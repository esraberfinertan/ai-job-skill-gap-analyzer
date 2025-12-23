AI Job Skill Gap Analyzer

CV → Job skill profile → Role match → Skill gap analysis → Visualization
A machine learning/NLP project that predicts which job cluster a CV belongs to, measures skill match percentage, identifies missing skills, and visualizes the results.

Project Overview

This project builds an end-to-end AI system that:
-Processes 22,000+ tech job postings
-Cleans and normalizes job text & skills
-Extracts TF-IDF features
-Clusters roles using K-Means
-Builds role-specific skill probability profiles
-Reads a CV (PDF or text)
-Detects which job cluster fits best
-Calculates role match vs skill gap %
-Visualizes strengths & weaknesses with radar & bar charts
Finished with a full-stack interactive web interface using Streamlit.

Key Features
AI/ML
✔ TF-IDF vectorization
✔ MiniBatch K-Means clustering
✔ High-dimensional job text analysis
✔ Skill frequency weighting

CV Analysis
✔ PDF parsing
✔ Skill detection
✔ Gap detection
✔ Match score

Visual Output
✔ Bar chart: match vs gap
✔ Radar chart: skill coverage profile

Web App
✔ Upload CV
✔ Run AI analysis
✔ Get score, roles, skills
✔ Instant feedback

Installation
git clone https://github.com/USERNAME/ai-job-skill-gap-analyzer.git
cd ai-job-skill-gap-analyzer
pip install -r requirements.txt

Run the Web App
streamlit run app.py

Tech Stack
Languages:
Python

Libraries:
pandas
scikit-learn
matplotlib
numpy
PyPDF2
streamlit

Roadmap:
 Add multilingual CV support
 Add LLM-based skill extraction
 Deploy web app online
 Add user authentication
 Add personalized upskilling track