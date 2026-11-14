# Smart Resume Matcher

A simple and effective Streamlit web application that helps recruiters and hiring managers quickly match candidate resumes with a Job Description.

The app extracts text from PDF resumes, calculates a match score using TF-IDF + Cosine Similarity, and identifies matched & missing skills.

---

## Features

- Upload multiple resume PDFs at once
- Paste any Job Description
- Automatic skill extraction (with whole-word matching)
- Ranked list of candidates based on match score
- Clear view of **Matched Skills** and **Missing Skills**
- Detailed candidate analysis
- Persistent history of previous analyses
- Download results as CSV
- Clean and user-friendly interface

---

## Tech Stack

- **Frontend / UI**: Streamlit
- **PDF Text Extraction**: PyMuPDF (`fitz`)
- **Text Matching**: scikit-learn (TF-IDF + Cosine Similarity)
- **Data Handling**: Pandas
- **Language**: Python

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/smart-resume-matcher.git
cd smart-resume-matcher