import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import joblib
from fpdf import FPDF

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_file as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load job descriptions dataset
def load_job_descriptions():
    return pd.read_csv("job_descriptions_cleaned.csv")

# Function to calculate similarity
def calculate_similarity(resume_text, job_descriptions):
    corpus = [resume_text] + job_descriptions['description'].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    job_descriptions['similarity'] = similarity_scores[0]
    return job_descriptions.sort_values(by='similarity', ascending=False)

# ML Model for Resume Scoring
def train_resume_scoring_model():
    df = pd.read_csv("resume_data.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['resume_text'])
    y = df['label']
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump((vectorizer, model), "resume_model.pkl")

# Load trained model
def load_resume_scoring_model():
    vectorizer, model = joblib.load("resume_model.pkl")
    return vectorizer, model

# Function to save results to PDF
def save_results_to_pdf(resume_name, matched_jobs, score):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Resume Analysis Report: {resume_name}", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Resume Score: {score:.2f}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Top Matched Jobs:", ln=True)
    pdf.ln(5)
    for index, row in matched_jobs.iterrows():
        pdf.cell(200, 10, txt=f"{row['title']} at {row['company']} (Similarity: {row['similarity']:.2f})", ln=True)
    pdf.output(f"{resume_name}_analysis.pdf")

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    
    # Custom Styling
    st.markdown("""
        <style>
            .main {
                background-color: #f8f9fa;
            }
            h1 {
                color: #0a75ad;
                text-align: center;
            }
            .uploadedFile {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>📄 AI Resume Analyzer & Job Matcher</h1>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_files = st.file_uploader("📂 Upload Your Resume(s) (PDF)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            resume_name = uploaded_file.name.replace(".pdf", "")
            resume_text = extract_text_from_pdf(uploaded_file)
            resume_text = preprocess_text(resume_text)

            # Layout for better presentation
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📑 Analysis for {resume_name}")
                job_descriptions = load_job_descriptions()
                matched_jobs = calculate_similarity(resume_text, job_descriptions)

                st.write("### 🔍 Top 5 Job Matches")
                st.dataframe(matched_jobs[['title', 'company', 'similarity']].head(5), use_container_width=True)

            with col2:
                vectorizer, model = load_resume_scoring_model()
                resume_vectorized = vectorizer.transform([resume_text])
                score = model.predict_proba(resume_vectorized)[0][1] * 100

                st.write("### 📊 Resume Score")
                st.progress(int(score))
                st.success(f"✅ Your resume score: {score:.2f}%")

                if score < 50:
                    st.warning("⚠️ Consider improving your resume with more relevant skills.")

            save_results_to_pdf(resume_name, matched_jobs.head(5), score)

            # Download Button
            with col1:
                with open(f"{resume_name}_analysis.pdf", "rb") as f:
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=f.read(),
                        file_name=f"{resume_name}_analysis.pdf",
                        mime="application/pdf",
                        help="Download a detailed report of your resume analysis."
                    )

if __name__ == "__main__":
    main()
