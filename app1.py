import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# --------------------------
# Paths for persistent storage
# --------------------------
DATA_DIR = Path("resume_matcher_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "analysis_history.json"

# --------------------------
# Expanded skill keywords
# --------------------------
SKILL_KEYWORDS = [
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "golang",
    "rust", "kotlin", "swift", "php", "ruby", "scala", "r", "matlab", "perl",
    "html", "css", "sql", "nosql", "bash", "shell scripting",

    # Data & ML / AI
    "machine learning", "deep learning", "data analysis", "data science",
    "data engineering", "nlp", "natural language processing", "computer vision",
    "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "huggingface",
    "transformers", "langchain", "llm", "large language models", "generative ai",
    "openai", "prompt engineering", "rag", "vector database", "embeddings",
    "feature engineering", "model deployment", "mlops", "statistics",
    "a/b testing", "experimentation",

    # BI & Visualization
    "excel", "power bi", "tableau", "looker", "qlik", "google data studio",
    "matplotlib", "seaborn", "plotly", "dash", "streamlit",

    # Cloud & DevOps
    "aws", "amazon web services", "azure", "google cloud", "gcp", "docker",
    "kubernetes", "k8s", "terraform", "ansible", "jenkins", "ci/cd",
    "github actions", "gitlab ci", "linux", "unix", "git", "github", "gitlab",
    "bitbucket", "nginx", "apache", "microservices", "serverless", "lambda",

    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra",
    "elasticsearch", "dynamodb", "snowflake", "bigquery", "redshift",
    "oracle", "sql server", "sqlite",

    # Backend / Frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "node.js", "express",
    "nestjs", "react", "angular", "vue", "next.js", "rest api", "graphql",
    "microservices architecture",

    # Soft Skills & Others
    "communication", "leadership", "teamwork", "collaboration", "problem solving",
    "critical thinking", "project management", "agile", "scrum", "kanban",
    "jira", "confluence", "stakeholder management", "presentation skills",
    "mentoring", "time management", "analytical skills", "attention to detail",

    # Domain / Other
    "etl", "data pipeline", "airflow", "dbt", "spark", "hadoop", "kafka",
    "cybersecurity", "network security", "penetration testing", "devops",
    "sre", "site reliability", "monitoring", "prometheus", "grafana",
    "selenium", "cypress", "pytest", "unit testing", "integration testing"
]

# --------------------------
# Helper functions
# --------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error reading PDF ({pdf_file.name}): {e}")
    return text

def calculate_match_score(resume_text, job_description):
    documents = [resume_text, job_description]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILL_KEYWORDS if skill in text]
    return sorted(list(set(found)))

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(entry):
    history = load_history()
    history.insert(0, entry)  # newest first
    # Keep only last 50 analyses to avoid huge files
    history = history[:50]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(
    page_title="Smart Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Smart Resume Matcher")
st.write("Upload resumes + Job Description → Get ranked matches with skill analysis. All results are saved automatically.")

# Sidebar
with st.sidebar:
    st.header("How to use")
    st.markdown("""
    1. Paste the **Job Description**
    2. Upload one or more **Resume PDFs**
    3. Click **Analyze Resumes**
    4. View ranked results + detailed skill gaps
    5. Results are **automatically saved** (like ChatGPT history)
    """)
    
    st.divider()
    st.subheader("📂 Saved Analyses")
    history = load_history()
    
    if history:
        for i, item in enumerate(history[:10]):  # show last 10
            label = f"{item['timestamp'][:16]} | {item['num_candidates']} candidates | JD: {item['job_title'][:30]}..."
            if st.button(label, key=f"hist_{i}"):
                st.session_state["loaded_history"] = item
                st.rerun()
    else:
        st.caption("No previous analyses yet.")

    if st.button("🗑️ Clear All History", type="secondary"):
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.success("History cleared!")
        st.rerun()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    job_description = st.text_area(
        "Paste the full Job Description here",
        height=250,
        placeholder="Paste JD text..."
    )
    job_title = st.text_input("Optional: Job Title (for history)", placeholder="e.g. Senior Data Scientist")

with col2:
    st.subheader("2. Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resume PDFs (multiple allowed)",
        type=["pdf"],
        accept_multiple_files=True
    )
    st.caption(f"{len(uploaded_files)} file(s) selected" if uploaded_files else "No files uploaded yet")

# Analyze button
analyze = st.button("🔍 Analyze Resumes", type="primary", use_container_width=True)

# --------------------------
# Analysis Logic
# --------------------------
if analyze:
    if not job_description.strip():
        st.warning("Please enter a Job Description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()

        job_skills = extract_skills(job_description)

        for idx, file in enumerate(uploaded_files):
            status.text(f"Processing: {file.name} ...")
            resume_text = extract_text_from_pdf(file)

            if not resume_text.strip():
                continue

            score = calculate_match_score(resume_text, job_description)
            resume_skills = extract_skills(resume_text)

            matched = sorted(list(set(resume_skills) & set(job_skills)))
            missing = sorted(list(set(job_skills) - set(resume_skills)))

            results.append({
                "Candidate": file.name,
                "Match Score (%)": score,
                "Matched Skills": ", ".join(matched) if matched else "None",
                "Missing Skills": ", ".join(missing) if missing else "None",
                "All Resume Skills": ", ".join(resume_skills),
                "Matched Count": len(matched),
                "Missing Count": len(missing)
            })
            progress.progress((idx + 1) / len(uploaded_files))

        status.empty()
        progress.empty()

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
            df.index = df.index + 1

            # Save to history (persistent)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "job_title": job_title or "Untitled JD",
                "job_description": job_description[:1500],  # store truncated for space
                "num_candidates": len(df),
                "results": df.to_dict(orient="records")
            }
            save_to_history(entry)

            st.success(f"✅ Analysis complete! {len(df)} candidates ranked and **saved to history**.")

            # Ranked table
            st.subheader("🏆 Ranked Candidates")
            st.dataframe(
                df[["Candidate", "Match Score (%)", "Matched Count", "Missing Count", "Matched Skills", "Missing Skills"]],
                use_container_width=True,
                height=400
            )

            # Download button
            csv = df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"resume_match_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            # Detailed expanders
            st.subheader("📋 Detailed Candidate View")
            for i, row in df.iterrows():
                with st.expander(f"#{i} — {row['Candidate']}  |  Score: **{row['Match Score (%)']}%**"):
                    st.markdown(f"**Matched Skills ({row['Matched Count']}):** {row['Matched Skills']}")
                    st.markdown(f"**Missing Skills ({row['Missing Count']}):** {row['Missing Skills']}")
                    st.markdown(f"**All skills found in resume:** {row['All Resume Skills']}")
        else:
            st.error("Could not extract text from any of the uploaded resumes.")

# --------------------------
# Show loaded history if any
# --------------------------
if "loaded_history" in st.session_state:
    item = st.session_state["loaded_history"]
    st.divider()
    st.subheader(f"📂 Loaded Previous Analysis — {item['timestamp'][:19]}")
    st.caption(f"Job: {item['job_title']} | Candidates: {item['num_candidates']}")

    hist_df = pd.DataFrame(item["results"])
    st.dataframe(hist_df, use_container_width=True)

    if st.button("Clear loaded history view"):
        del st.session_state["loaded_history"]
        st.rerun()

# Footer
st.markdown("---")
st.caption("Smart Resume Matcher • Persistent history saved locally • Built with Streamlit + PyMuPDF + scikit-learn")