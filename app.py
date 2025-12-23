import ast
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------
# Data & model loading
# ---------------------------

@st.cache_resource
def load_data_and_model():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    processed_dir = data_dir / "processed"
    csv_path = processed_dir / "jobs_clean.csv"

    df = pd.read_csv(csv_path)

    # TF-IDF + clustering
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )
    X = vectorizer.fit_transform(df["jobdescription_clean"])

    k = 10
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    df["cluster"] = clusters

    # skills_clean: string -> list
    df_skills = df.dropna(subset=["skills_clean"]).copy()
    df_skills["skills_clean"] = df_skills["skills_clean"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # skill cleaning
    def clean_skill(skill: str):
        skill = skill.lower().strip()
        remove_list = ["see below", "please see job description", "haha"]
        for bad in remove_list:
            if bad in skill:
            # discard noisy entries
                return None
        skill = re.sub(r"[^a-z0-9#+. ]", "", skill)
        skill = skill.strip(" .,+-")
        if len(skill) == 0:
            return None
        return skill

    def rebuild_skills_list(skill_list):
        if not isinstance(skill_list, list):
            return None
        cleaned = []
        for sk in skill_list:
            sk2 = clean_skill(sk)
            if sk2:
                cleaned.append(sk2)
        return cleaned if cleaned else None

    df_skills["skills_filtered"] = df_skills["skills_clean"].apply(
        rebuild_skills_list
    )

    # cluster skill profiles
    cluster_skill_profiles = {}
    for c in sorted(df_skills["cluster"].unique()):
        skills = []
        for row in df_skills[df_skills["cluster"] == c]["skills_filtered"]:
            if isinstance(row, list):
                skills.extend(row)
        cluster_skill_profiles[c] = Counter(skills).most_common(20)

    # skill weights (frequency-based)
    skill_weights = {}
    for c, skills in cluster_skill_profiles.items():
        if skills:
            max_freq = max(freq for skill, freq in skills)
        else:
            max_freq = 1
        skill_weights[c] = {skill: freq / max_freq for skill, freq in skills}

    return df, vectorizer, kmeans, cluster_skill_profiles, skill_weights


df, vectorizer, kmeans, cluster_skill_profiles, skill_weights = load_data_and_model()


# ---------------------------
# Text & CV utilities
# ---------------------------

def clean_text_for_cv(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9#+.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf_text(pdf_file) -> str:
    # pdf_file: UploadedFile from streamlit or path-like
    if hasattr(pdf_file, "read"):
        reader = PyPDF2.PdfReader(pdf_file)
    else:
        with open(pdf_file, "rb") as f:
            reader = PyPDF2.PdfReader(f)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def analyze_cv(cv_text: str, top_n_skills: int = 15):
    cv_clean = clean_text_for_cv(cv_text)
    cv_vec = vectorizer.transform([cv_clean])

    cluster_id = kmeans.predict(cv_vec)[0]

    titles = df[df["cluster"] == cluster_id]["jobtitle"]
    top_titles = titles.value_counts().head(5)

    cluster_skills = [
        s for s, c in cluster_skill_profiles[cluster_id][:top_n_skills]
    ]

    present_skills = [s for s in cluster_skills if s in cv_clean]
    missing_skills = [s for s in cluster_skills if s not in present_skills]

    weights = skill_weights[cluster_id]
    present_score = sum(weights.get(skill, 0) for skill in present_skills)
    total_score = sum(weights.get(skill, 0) for skill in cluster_skills)
    match_score = (present_score / total_score) if total_score > 0 else 0.0
    gap_score = 1 - match_score

    return {
        "cluster_id": int(cluster_id),
        "top_titles": top_titles,
        "cluster_top_skills": cluster_skills,
        "present_skills": present_skills,
        "missing_skills": missing_skills,
        "match_score": float(match_score),
        "gap_score": float(gap_score),
    }


def make_match_gap_figure(match_score: float):
    labels = ["Match %", "Gap %"]
    values = [match_score * 100, (1 - match_score) * 100]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values)
    ax.set_title("CV Role Fit Analysis (%)")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    return fig


def make_radar_figure(expected, present, cluster_id: int):
    skills = expected
    values = [1 if s in present else 0 for s in skills]

    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills, fontsize=9)
    ax.set_title(f"CV Skill Coverage Profile — Cluster {cluster_id}")
    return fig


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AI Job Skill Gap Analyzer", layout="wide")

st.title("🧠 AI Job Skill Gap Analyzer")
st.markdown(
    """
CV'ini gerçek iş ilanlarına göre analiz eden bir NLP aracı.  
Yapılanlar:
- İş ilanlarını TF-IDF + KMeans ile rol kümelerine ayırıyoruz
- Her rol için en yaygın skill profilini çıkarıyoruz
- CV'ni bu rol kümelerine göre skorlayıp **match / gap** yüzdesi hesaplıyoruz
"""
)

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("📄 CV PDF yükle", type=["pdf"])
    st.markdown("veya")
    manual_text = st.text_area(
        "✍️ CV metnini buraya yapıştır",
        height=200,
        placeholder="Özet, deneyimler, teknolojiler...",
    )

    analyze_button = st.button("🚀 Analiz Et")

with col_right:
    st.info(
        "İpucu: En iyi sonuç için İngilizce CV / skill listesi kullanman daha sağlıklı olur."
    )

if analyze_button:
    if uploaded_file is not None:
        cv_text = read_pdf_text(uploaded_file)
    elif manual_text.strip():
        cv_text = manual_text
    else:
        st.warning("Lütfen bir PDF yükle veya metin gir 💛")
        st.stop()

    if not cv_text.strip():
        st.error("PDF'den metin okunamadı. (Scanned / image-based olabilir)")
        st.stop()

    result = analyze_cv(cv_text, top_n_skills=15)

    st.subheader("🎯 Sonuç Özeti")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Role Match Score",
            f"{result['match_score']*100:.1f} %",
        )
    with col2:
        st.metric(
            "Skill Gap Score",
            f"{result['gap_score']*100:.1f} %",
        )

    st.markdown("**Tahmin edilen rol kümesi (Cluster)**: `{}`".format(result["cluster_id"]))

    st.markdown("**Bu rolde en sık görülen iş unvanları:**")
    st.write(result["top_titles"])

    st.markdown("**Beklenen skill profili:**")
    st.write(result["cluster_top_skills"])

    st.markdown("**CV'de bulunan skill'ler:**")
    st.write(result["present_skills"])

    st.markdown("**Eksik olan kritik skill'ler:**")
    st.write(result["missing_skills"])

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_bar = make_match_gap_figure(result["match_score"])
        st.pyplot(fig_bar)

    with chart_col2:
        fig_radar = make_radar_figure(
            result["cluster_top_skills"],
            result["present_skills"],
            result["cluster_id"],
        )
        st.pyplot(fig_radar)
