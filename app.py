import os
import re
from io import BytesIO

import streamlit as st
from google import genai
from pypdf import PdfReader
from docx import Document

MODEL_NAME = "gemini-3.8-flash"
MAX_FILE_SIZE_MB = 10

st.set_page_config(page_title="Resume ATS Checker", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #f7faf8;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#123c32,#218568); color: white; margin-bottom: 1rem;}
    .hero h1 {color: white; margin: 0;}
    .hero p {color: #e7fff5; margin-bottom: 0;}
    </style>
    <div class="hero">
        <h1>📄 Fast Resume ATS Checker</h1>
        <p>Upload your resume, add the target job description, and get instant AI-powered improvement tips.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def extract_resume_text(uploaded_file) -> str:
    """Fast text extraction from PDF or DOCX."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file_name.endswith(".docx"):
        document = Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
    else:
        raise ValueError("Please upload a PDF or DOCX file.")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    
    if not text:
        raise ValueError("No text found. If scanned, please use a text-based file.")
    
    # Cap text length to ensure super fast API processing
    return text[:4000]


def calculate_ats_estimate(resume_text: str, job_description: str) -> dict:
    """Fast rule-based estimate."""
    text = resume_text.lower()
    sections = {
        "Contact details": any(x in text for x in ["email", "@", "phone", "linkedin"]),
        "Education": any(x in text for x in ["education", "university", "bachelor", "degree"]),
        "Work experience": any(x in text for x in ["experience", "employment", "internship", "work history"]),
        "Skills": "skills" in text or "technical skills" in text,
        "Projects or achievements": any(x in text for x in ["projects", "achievements", "certifications"]),
    }

    word_count = len(re.findall(r"\b\w+\b", resume_text))
    length_score = 10 if 250 <= word_count <= 1000 else 6
    section_score = round((sum(sections.values()) / len(sections)) * 35)
    contact_score = 10 if sections["Contact details"] else 0

    matched_keywords, missing_keywords = [], []
    if job_description.strip():
        stop_words = {"the", "and", "for", "with", "you", "your", "are", "will", "this", "from", "have", "that", "work"}
        job_words = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]{2,}", job_description.lower())
        keywords = list(dict.fromkeys([w for w in job_words if w not in stop_words]))[:25]
        matched_keywords = [w for w in keywords if w in text]
        missing_keywords = [w for w in keywords if w not in text]
        keyword_score = round((len(matched_keywords) / max(len(keywords), 1)) * 45)
    else:
        keyword_score = 25

    score = min(100, max(0, section_score + contact_score + length_score + keyword_score))
    return {
        "score": score,
        "sections": sections,
        "word_count": word_count,
        "matched_keywords": matched_keywords[:10],
        "missing_keywords": missing_keywords[:10],
        "job_description_provided": bool(job_description.strip()),
    }


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "")


def get_ai_review_fast(resume_text: str, job_description: str, score_data: dict) -> str:
    """Super fast direct API call without streaming latency."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Gemini API Key is missing.")

    client = genai.Client(api_key=api_key)
    jd = job_description.strip()[:2000] or "General resume optimization."
    
    prompt = f"""
You are an expert HR Manager and ATS Evaluator.
Analyze this resume against the target job description rapidly.

Score: {score_data['score']}/100
Word Count: {score_data['word_count']}

=== JOB DESCRIPTION ===
{jd}

=== RESUME TEXT ===
{resume_text}

Provide concise output in clean Markdown with:
1. **Overall Fit**: (2 short sentences)
2. **Top 3 Critical Improvements**: (Specific bullet points)
3. **Missing High-Value Keywords**: (Comma-separated list)
4. **Key Formatting Tip**: (1 sentence)
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a **PDF** or **DOCX** resume.\n"
        "2. Paste the target job description.\n"
        "3. Click **Analyze resume**."
    )

left, right = st.columns([1, 1])
with left:
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
with right:
    job_description = st.text_area("Target job description", height=180, placeholder="Paste JD here...")

analyze = st.button("Analyze resume", type="primary", use_container_width=True)

if analyze:
    if uploaded_file is None:
        st.error("Please upload your resume first.")
    else:
        try:
            with st.spinner("Extracting text and running analysis..."):
                resume_text = extract_resume_text(uploaded_file)
                score_data = calculate_ats_estimate(resume_text, job_description)
                
            st.session_state["resume_text"] = resume_text
            st.session_state["score_data"] = score_data
            st.session_state["job_description"] = job_description
            st.session_state.pop("ai_review", None)
        except Exception as exc:
            st.error(str(exc))

if "score_data" in st.session_state:
    data = st.session_state["score_data"]
    st.divider()
    st.subheader("Your ATS-readiness estimate")
    metric_col, word_col, section_col = st.columns(3)
    metric_col.metric("Estimated score", f"{data['score']}/100")
    word_col.metric("Resume word count", data["word_count"])
    section_col.metric("Detected sections", f"{sum(data['sections'].values())}/5")
    st.progress(data["score"] / 100)

    if st.button("⚡ Fast Generate Gemini Report", type="primary"):
        try:
            with st.spinner("Generating fast report with Gemini..."):
                report = get_ai_review_fast(
                    st.session_state["resume_text"],
                    st.session_state.get("job_description", ""),
                    data,
                )
                st.session_state["ai_review"] = report
        except Exception as exc:
            st.error(f"Could not generate report: {exc}")

    if st.session_state.get("ai_review"):
        st.divider()
        st.subheader("Gemini Improvement Report")
        st.markdown(st.session_state["ai_review"])
        st.download_button(
            "Download Report (.md)",
            data=st.session_state["ai_review"],
            file_name="resume_report.md",
            mime="text/markdown",
        )
