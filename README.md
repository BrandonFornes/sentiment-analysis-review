# Multi-Class Sentiment Analysis Platform

An end-to-end NLP and Machine Learning application that classifies text sentiment into three categories: **Negative**, **Neutral**, or **Positive**. 

Features an optimized Linear SVM model, an interactive **Streamlit** UI, and full **Docker** containerization.

---

## Live Demo
**https://huggingface.co/spaces/BrandonFornes/sentiment-analysis-app**

## Quick Start

### Option 1: Using Docker (Recommended)
1. **Build the image:**
   ```bash
   docker build -t sentiment-app .
   
2. **Run the container**:
   ```bash
   docker run -p 8501:8501 sentiment-app 
Access the app at http://localhost:8501.

Option 2: Native Python Setup
**Install dependencies**:
   pip install -r requirements.txt
Run the app:

streamlit run app.py
Tech Stack
Core: Python 3.11, Scikit-Learn, Pandas, NumPy

NLP: NLTK (PorterStemmer, Stopwords)

Deployment: Streamlit, Docker, Hugging Face Spaces
