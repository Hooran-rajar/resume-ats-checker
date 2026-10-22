
import os
import re
from io import BytesIO

import streamlit as st
from google import genai
from pypdf import PdfReader
from docx import Document

MODEL_NAME = "gemini-3.8-flash"
MAX_FILE_SIZE_MB = 10

st.set_page_config(
    page_title="Resume ATS Checker",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {background-color: #f7faf8;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg,#123c32,#218568);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin: 0;}
    .hero p {color: #e7fff5; margin-bottom: 0;}
    </style>
    <div class="hero">
        <h1>📄 Fast Resume ATS Checker</h1>
        <p>Upload your resume, add a job description, and get
        ATS estimates with Gemini AI or basic suggestions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def extract_resume_text(uploaded_file):
    """Extract text from PDF or DOCX files."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    elif file_name.endswith(".docx"):
        document = Document(BytesIO(file_bytes))
        text = "\n".join(
            p.text for p in document.paragraphs if p.text.strip()
        )
    else:
        raise ValueError("Please upload a PDF or DOCX file.")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError(
            "No text found. If your PDF is scanned, use a text-based file."
        )

    return text[:4000]


def calculate_ats_estimate(resume_text, job_description):
    """Calculate a simple rule-based ATS-readiness estimate."""
    text = resume_text.lower()

    sections = {
        "Contact details": any(
            x in text for x in ["email", "@", "phone", "linkedin"]
        ),
        "Education": any(
            x in text for x in ["education", "university", "bachelor", "degree"]
        ),
        "Work experience": any(
            x in text
            for x in ["experience", "employment", "internship", "work history"]
        ),
        "Skills": any(x in text for x in ["skills", "technical skills"]),
        "Projects or achievements": any(
            x in text
            for x in ["projects", "achievements", "certifications"]
        ),
    }

    word_count = len(re.findall(r"\b\w+\b", resume_text))
    length_score = 10 if 250 <= word_count <= 1000 else 6
    section_score = round(
        (sum(sections.values()) / len(sections)) * 35
    )
    contact_score = 10 if sections["Contact details"] else 0

    matched_keywords = []
    missing_keywords = []

    if job_description.strip():
        stop_words = {
            "the", "and", "for", "with", "you", "your", "are",
            "will", "this", "from", "have", "that", "work",
        }

        job_words = re.findall(
            r"[a-zA-Z][a-zA-Z+#.\-]{2,}",
            job_description.lower(),
        )
        keywords = list(
            dict.fromkeys(
                word for word in job_words if word not in stop_words
            )
        )[:25]

        matched_keywords = [
            word for word in keywords if word in text
        ]
        missing_keywords = [
            word for word in keywords if word not in text
        ]

        keyword_score = round(
            len(matched_keywords) / max(len(keywords), 1) * 45
        )
    else:
        keyword_score = 25

    score = min(
        100,
        max(0, section_score + contact_score + length_score + keyword_score),
    )

    return {
        "score": score,
        "sections": sections,
        "word_count": word_count,
        "matched_keywords": matched_keywords[:10],
        "missing_keywords": missing_keywords[:10],
        "job_description_provided": bool(job_description.strip()),
    }


def get_api_key():
    """Read the API key from Streamlit Secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GEMINI_API_KEY", "")


def build_fallback_report(score_data, job_description):
    """Generate basic suggestions locally, without Gemini."""
    tips = []
    sections = score_data["sections"]

    missing_sections = [
        name for name, present in sections.items() if not present
    ]

    if missing_sections:
        tips.append(
            "Add clear section headings where relevant: "
            + ", ".join(missing_sections)
            + "."
        )

    word_count = score_data["word_count"]

    if word_count < 250:
        tips.append(
            "Your extracted resume text is short. If relevant, add "
            "details about education, projects, skills, or experience."
        )
    elif word_count > 1000:
        tips.append(
            "Review your resume for repetition and prioritize information "
            "relevant to the target role."
        )
    else:
        tips.append(
            "Use concise bullet points and action verbs. Include measurable "
            "achievements when you can support them accurately."
        )

    if job_description.strip():
        missing = score_data["missing_keywords"]
        matched = score_data["matched_keywords"]

        if missing:
            tips.append(
                "Review these job-description terms and include only those "
                "that truthfully match your experience: "
                + ", ".join(missing)
                + "."
            )

        if matched:
            tips.append(
                "Relevant terms detected in your resume: "
                + ", ".join(matched)
                + "."
            )
    else:
        tips.append(
            "Paste the target job description to compare relevant keywords."
        )

    tips.append(
        "Use standard headings, consistent dates, readable formatting, "
        "and avoid putting essential information only inside images."
    )

    lines = [
        "# Basic Resume Improvement Suggestions",
        "",
        f"**Rule-based ATS-readiness estimate:** "
        f"{score_data['score']}/100",
        f"**Extracted word count:** {word_count}",
        "",
        "> This is a basic rule-based estimate, not an official ATS score.",
        "",
        "## Suggested improvements",
    ]

    lines.extend(f"- {tip}" for tip in tips)
    return "\n".join(lines)


def get_ai_review(resume_text, job_description, score_data):
    """Request the AI-generated report from Gemini."""
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check Streamlit Secrets."
        )

    client = genai.Client(api_key=api_key)

    jd = job_description.strip()[:2000]
    if not jd:
        jd = "General resume optimization."

    prompt = f"""
You are an expert HR manager and ATS evaluator.

Analyze the resume against the target job description.
Do not invent qualifications or experience.

Rule-based estimated score: {score_data['score']}/100
Extracted word count: {score_data['word_count']}

JOB DESCRIPTION:
{jd}

RESUME:
{resume_text}

Provide a concise Markdown report with:
1. Overall fit: two short sentences.
2. Top 3 critical improvements: specific bullet points.
3. Missing high-value keywords: list only relevant terms.
4. One key formatting tip.

Clearly distinguish suggestions from facts.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not getattr(response, "text", None):
        raise RuntimeError("Gemini returned an empty report.")

    return response.text


with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a PDF or DOCX resume.\n"
        "2. Paste the target job description.\n"
        "3. Click Analyze resume.\n"
        "4. Generate an AI report or basic suggestions."
    )
    st.caption(f"Gemini model: {MODEL_NAME}")


left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or DOCX, max 10 MB)",
        type=["pdf", "docx"],
    )

with right:
    job_description = st.text_area(
        "Target job description",
        height=180,
        placeholder="Paste the job description here...",
    )


analyze = st.button(
    "Analyze resume",
    type="primary",
    use_container_width=True,
)


if analyze:
    if uploaded_file is None:
        st.error("Please upload your resume first.")

    elif uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Please upload a file smaller than {MAX_FILE_SIZE_MB} MB.")

    else:
        try:
            with st.spinner("Extracting text and analyzing resume..."):
                resume_text = extract_resume_text(uploaded_file)
                score_data = calculate_ats_estimate(
                    resume_text, job_description
                )

            st.session_state["resume_text"] = resume_text
            st.session_state["score_data"] = score_data
            st.session_state["job_description"] = job_description

            st.session_state.pop("ai_review", None)
            st.session_state.pop("fallback_report", None)

        except Exception as exc:
            st.error(f"Could not analyze resume: {exc}")


if "score_data" in st.session_state:
    data = st.session_state["score_data"]

    st.divider()
    st.subheader("Your ATS-readiness estimate")

    metric_col, word_col, section_col = st.columns(3)

    metric_col.metric("Estimated score", f"{data['score']}/100")
    word_col.metric("Resume word count", data["word_count"])
    section_col.metric(
        "Detected sections",
        f"{sum(data['sections'].values())}/5",
    )

    st.progress(data["score"] / 100)

    st.caption(
        "This is a basic rule-based estimate, not an official ATS score "
        "or a guarantee of hiring outcomes."
    )

    with st.expander("Detected resume sections"):
        for section, present in data["sections"].items():
            st.write(
                f"{'✅' if present else '⚠️'} {section}"
            )

    if data["job_description_provided"]:
        st.markdown("**Job-description keyword comparison**")
        keyword_col1, keyword_col2 = st.columns(2)

        with keyword_col1:
            st.markdown("**Matched terms**")
            st.write(
                ", ".join(data["matched_keywords"])
                if data["matched_keywords"]
                else "No matching terms detected."
            )

        with keyword_col2:
            st.markdown("**Terms to review**")
            st.write(
                ", ".join(data["missing_keywords"])
                if data["missing_keywords"]
                else "No missing terms detected by this simple check."
            )

    st.divider()
    st.subheader("Resume improvement reports")

    st.write(
        "Gemini generates an AI report when its API is available. "
        "Basic suggestions work locally without Gemini."
    )

    ai_col, fallback_col = st.columns(2)

    with ai_col:
        generate_ai = st.button(
            "⚡ Generate Gemini AI Report",
            type="primary",
            use_container_width=True,
        )

    with fallback_col:
        generate_fallback = st.button(
            "📝 Generate Basic Suggestions",
            use_container_width=True,
        )

    if generate_ai:
        try:
            with st.spinner("Generating Gemini report..."):
                report = get_ai_review(
                    st.session_state["resume_text"],
                    st.session_state.get("job_description", ""),
                    data,
                )

            st.session_state["ai_review"] = report
            st.session_state.pop("fallback_report", None)

        except Exception as exc:
            error_text = str(exc)
            error_lower = error_text.lower()

            is_quota_or_temporary = any(
                term in error_lower
                for term in (
                    "429",
                    "resource_exhausted",
                    "quota",
                    "rate limit",
                    "503",
                    "unavailable",
                    "resource exhausted",
                )
            )

            if is_quota_or_temporary:
                st.warning(
                    "Gemini's quota may be exhausted or the service may be "
                    "temporarily unavailable. Your ATS score is still "
                    "available. Generate basic suggestions below."
                )
            else:
                st.error(f"Could not generate Gemini report: {error_text}")

            st.session_state.pop("ai_review", None)

    if generate_fallback:
        st.session_state["fallback_report"] = build_fallback_report(
            data,
            st.session_state.get("job_description", ""),
        )
        st.session_state.pop("ai_review", None)

    if st.session_state.get("ai_review"):
        st.divider()
        st.subheader("Gemini AI Improvement Report")
        st.markdown(st.session_state["ai_review"])

        st.download_button(
            "Download Gemini Report (.md)",
            data=st.session_state["ai_review"],
            file_name="resume_gemini_report.md",
            mime="text/markdown",
        )

    if st.session_state.get("fallback_report"):
        st.divider()
        st.subheader("Basic Improvement Suggestions")
        st.markdown(st.session_state["fallback_report"])

        st.download_button(
            "Download Basic Suggestions (.md)",
            data=st.session_state["fallback_report"],
            file_name="resume_basic_suggestions.md",
            mime="text/markdown",
        )
