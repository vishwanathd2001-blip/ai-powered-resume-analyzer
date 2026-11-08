
import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

SKILLS = [
    "python", "java", "sql", "machine learning",
    "deep learning", "artificial intelligence",
    "data analysis", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "streamlit", "excel", "power bi",
    "tableau", "html", "css", "javascript",
    "communication", "problem solving",
    "electrical engineering", "matlab",
    "autocad", "power systems"
]

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return " ".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    return uploaded_file.getvalue().decode(
        "utf-8", errors="ignore"
    )

def find_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)
    return found

def calculate_similarity(resume, job):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

st.title("📄 AI-Powered Resume Analyzer")
st.write(
    "Upload your resume and compare your skills "
    "with a job description."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or TXT)",
    type=["pdf", "txt"]
)

job_description = st.text_area(
    "Paste the job description here",
    height=200,
    placeholder="Enter the job requirements..."
)

if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:
        st.error("Please upload your resume.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        try:
            resume_text = extract_text(uploaded_file)

            if not resume_text.strip():
                st.error("No readable text found in the resume.")
            else:
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)

                matched = [
                    skill for skill in job_skills
                    if skill in resume_skills
                ]

                missing = [
                    skill for skill in job_skills
                    if skill not in resume_skills
                ]

                similarity = calculate_similarity(
                    resume_text, job_description
                )

                coverage = (
                    len(matched) / len(job_skills) * 100
                    if job_skills else 0
                )

                final_score = round(
                    similarity * 0.7 + coverage * 0.3, 2
                )

                st.divider()
                st.header("Analysis Results")

                col1, col2, col3 = st.columns(3)

                col1.metric("Resume Match", f"{final_score}%")
                col2.metric("Matched Skills", len(matched))
                col3.metric("Missing Skills", len(missing))

                st.subheader("Matched Skills")
                if matched:
                    st.success(", ".join(matched))
                else:
                    st.info("No matching skills detected.")

                st.subheader("Skills to Improve")
                if missing:
                    st.warning(", ".join(missing))
                else:
                    st.success("All detected job skills are present!")

                st.subheader("Detailed Skill Report")

                report = pd.DataFrame({
                    "Skill": matched + missing,
                    "Status": (
                        ["Matched"] * len(matched)
                        + ["Missing"] * len(missing)
                    )
                })

                st.dataframe(report, use_container_width=True)

                st.subheader("Recommendation")

                if final_score >= 70:
                    st.success(
                        "Your resume shows a good match. "
                        "Review the job requirements before applying."
                    )
                elif final_score >= 40:
                    st.info(
                        "Your resume has a moderate match. "
                        "Improve the missing skills where relevant."
                    )
                else:
                    st.warning(
                        "Your resume has a low match. "
                        "Consider developing relevant skills."
                    )

                st.caption(
                    "Note: This is an educational TF-IDF-based "
                    "matching tool, not an automated hiring decision."
                )

        except Exception as error:
            st.error(f"Unable to analyze the resume: {error}")
