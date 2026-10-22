                if is_temporary:
                    raise RuntimeError(
                        "Gemini is busy or temporarily unavailable. Please try again in a little while."
                    ) from exc
                raise RuntimeError(f"Gemini request failed: {message}") from exc

            time.sleep(2 ** (attempt + 1))  # Wait 2 seconds, then 4 seconds.

    raise RuntimeError("Gemini did not return a response. Please try again.")


with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a text-based **PDF** or **DOCX** resume.\n"
        "2. Paste the job description for a more useful keyword comparison.\n"
        "3. Select **Analyze resume**.\n\n"
        "**Privacy tip:** Upload only a resume you are comfortable processing with an AI service. The app does not save uploaded files to a database."
    )
    st.caption("Supported files: PDF and DOCX · Maximum size: 5 MB")

left, right = st.columns([1, 1])
with left:
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx"],
        help="PDFs must contain selectable text; scanned-image PDFs are not supported in this version.",
    )
with right:
    job_description = st.text_area(
        "Target job description (recommended)",
        height=180,
        placeholder="Paste the job description here to compare relevant skills and keywords...",
    )

analyze = st.button("Analyze resume", type="primary", use_container_width=True)

if analyze:
    if uploaded_file is None:
        st.error("Please upload your resume first.")
    elif uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"The file is too large. Please upload a file smaller than {MAX_FILE_SIZE_MB} MB.")
    else:
        try:
            with st.spinner("Reading your resume..."):
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
    st.caption(
        "The estimate is based on basic section checks, resume length, and simple job-keyword matching. "
        "It is not a validated prediction of an employer's ATS result."
    )

    st.subheader("Section checks")
    section_cols = st.columns(3)
    for index, (section, found) in enumerate(data["sections"].items()):
        with section_cols[index % 3]:
            st.write(("✅ " if found else "⚠️ ") + section)

    if data["job_description_provided"]:
        key_col1, key_col2 = st.columns(2)
        with key_col1:
            st.markdown("**Matched job-description terms**")
            st.write(", ".join(data["matched_keywords"]) if data["matched_keywords"] else "No simple matches detected.")
        with key_col2:
            st.markdown("**Terms not detected in resume**")
            st.write(", ".join(data["missing_keywords"]) if data["missing_keywords"] else "No missing terms detected by the simple matcher.")
        st.caption("Keyword matching is literal and may miss synonyms or multi-word skills.")
    else:
        st.warning("Add a target job description next time for a more relevant keyword comparison.")

    if st.button("Generate Gemini improvement report", type="primary"):
        try:
            st.session_state.pop("ai_review", None)
            st.subheader("Gemini improvement report")
            with st.spinner("Connecting to Gemini..."):
                report_stream = get_ai_review(
                    st.session_state["resume_text"],
                    st.session_state.get("job_description", ""),
                    data,
                )
                st.session_state["ai_review"] = st.write_stream(report_stream)
            if not st.session_state["ai_review"].strip():
                st.warning("Gemini returned an empty response. Please try again.")
        except Exception as exc:
            st.error(f"Could not generate the AI report: {exc}")

    if st.session_state.get("ai_review"):
        st.divider()
        st.subheader("Gemini improvement report")
        st.markdown(st.session_state["ai_review"])
        st.download_button(
            "Download AI report (.md)",
            data=st.session_state["ai_review"],
            file_name="resume_improvement_report.md",
            mime="text/markdown",
        )

st.divider()
st.caption("Built with Streamlit and Google's Gemini API. AI output ca
