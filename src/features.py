import re
import ast
from collections import Counter
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np

def plot_skill_radar(expected, present, cluster_id):
    skills = sorted(expected)
    values = [1 if s in present else 0 for s in skills]

    N = len(skills)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    values += values[:1]
    angles = np.concatenate((angles, [angles[0]]))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, marker="o")
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills)

    plt.title(f"CV Skill Coverage Profile — Cluster {cluster_id}", fontsize=16)
    plt.show()
    
def parse_skills_column(df):
    """
    Converts string skill lists into Python list objects.
    Drops rows with missing skill data.
    """
    df_skills = df.dropna(subset=["skills_clean"]).copy()
    
    df_skills["skills_clean"] = df_skills["skills_clean"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    return df_skills


def clean_skill(skill: str) -> Union[str, None]:
    """
    Normalizes individual skill token.
    Keeps characters: a-z, 0-9, #, +, .
    Removes noise patterns.
    """
    skill = skill.lower().strip()
    
    remove_list = [
        "see below",
        "please see job description",
        "haha"
    ]
    
    for bad in remove_list:
        if bad in skill:
            return None
    
    skill = re.sub(r"[^a-z0-9#+. ]", "", skill)
    skill = skill.strip(" .,+-")
    
    if len(skill) == 0:
        return None
    
    return skill


def rebuild_skills_list(skill_list: Union[List[str], None]) -> Union[List[str], None]:
    """
    Builds cleaned skill list for each job row.
    """
    if not isinstance(skill_list, list):
        return None
    
    cleaned: List[str] = []
    
    for skill in skill_list:
        cleaned_skill = clean_skill(skill)
        if cleaned_skill:
            cleaned.append(cleaned_skill)
    
    return cleaned if cleaned else None


def build_skill_profiles(df):
    """
    Produces top skills per cluster.
    """
    profiles = {}

    for c in sorted(df["cluster"].unique()):
        skills = []
        for row in df[df["cluster"] == c]["skills_filtered"]:
            if isinstance(row, list):
                skills.extend(row)
        profiles[c] = Counter(skills).most_common(20)
    
    return profiles
