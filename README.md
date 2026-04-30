# Resume Shortlisting App

A Streamlit web app that uses ML + NLP to automatically rank and shortlist resumes against a job description. Upload multiple resumes, paste a job description, and get a ranked shortlist instantly.


## Features
- Upload multiple PDF/DOCX resumes
- Paste any job description
- NLP-based similarity scoring (TF-IDF / sentence embeddings)
- Ranked output with match percentage per candidate

## Tech Stack
- Python, scikit-learn, NLTK / spaCy
- Streamlit
- PyPDF2 / python-docx for resume parsing

## How to Run
```bash
git clone https://github.com/saloni365-ops/Resume-Shortlisting-App
cd Resume-Shortlisting-App
pip install -r requirements.txt
streamlit run app.py
```

## How it Works
1. Resumes are parsed and cleaned (remove stopwords, lemmatize)
2. Job description and resumes are vectorized using TF-IDF
3. Cosine similarity scores each resume against the JD
4. Results ranked from highest to lowest match
