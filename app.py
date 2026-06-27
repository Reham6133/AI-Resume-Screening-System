from skills import skills
import streamlit as st
import easyocr
from PIL import Image
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)
st.sidebar.title("📄 AI Resume Screening")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠 Technologies Used")

st.sidebar.write("• EasyOCR")
st.sidebar.write("• Sentence-BERT")
st.sidebar.write("• Cosine Similarity")
st.sidebar.write("• Streamlit")
st.sidebar.write("• Python")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a resume image and compare it with a job description using AI."
)
def extract_skills(text, skills_db):
    found_skills = []
    for skill in skills_db:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills
def semantic_match(resume_skills, job_skills):

    semantic_matches = []

    for job_skill in job_skills:

        for resume_skill in resume_skills:

            emb1 = model.encode(job_skill)
            emb2 = model.encode(resume_skill)

            similarity = util.cos_sim(emb1, emb2).item()

            if similarity >= 0.5:
                semantic_matches.append(job_skill)
                break

    return semantic_matches
def calculate_match_score(semantic_matches, job_skills):

    score = (len(semantic_matches) / len(job_skills)) * 100

    return round(score, 2)
def get_missing_skills(job_skills, semantic_matches):

    missing_skills = []

    for skill in job_skills:
        if skill not in semantic_matches:
            missing_skills.append(skill)

    return missing_skills       
st.title("📄 AI Resume Screening System")
st.markdown(
    "### Upload a resume image and compare it with a job description using AI-powered semantic matching."
)
left_col, right_col = st.columns(2)

with left_col:
    uploaded_file = st.file_uploader(
        "📤 Upload Resume Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Resume", use_container_width=True)

        result = reader.readtext(image, detail=0)

        extracted_text = " ".join(result)

        with st.expander("📄 View Extracted Resume Text"):
            st.write(extracted_text)

with right_col:
    resume_text = st.text_area(
        "Resume Text",
        value=extracted_text if uploaded_file is not None else "",
        height=250
    )

    job_description = st.text_area(
        "Job Description",
        height=250
    )
if st.button("🔍 Analyze Resume"):

    resume_skills = extract_skills(resume_text, skills)

    job_skills = extract_skills(job_description, skills)

    if len(job_skills) == 0:
        st.error("Please enter a job description.")

    else:
        semantic_matches = semantic_match(
            resume_skills,
            job_skills
        )

        match_score = calculate_match_score(
            semantic_matches,
            job_skills
        )

        missing_skills = get_missing_skills(
            job_skills,
            semantic_matches
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Match Score", f"{match_score}%")

        with col2:
            st.metric("Matched Skills", len(semantic_matches))

        st.progress(match_score / 100)
        if match_score >= 80:
            st.success("🎉 Excellent Match! This resume closely matches the job description.")

        elif match_score >= 60:
            st.info("👍 Good Match! The candidate meets most of the requirements.")

        else:
            st.error("⚠️ Low Match! The resume is missing several important skills.")
        st.success("Matched Skills: " + ", ".join(semantic_matches))

        if missing_skills:
         st.warning(
            "Missing Skills: " + ", ".join(missing_skills)
         )

        