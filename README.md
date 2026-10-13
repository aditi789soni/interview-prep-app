# 🎯 AI Technical & Behavioral Interview Simulator

A full-stack, single-file Streamlit application that acts as an intelligent interviewer and evaluator powered by **Google Gemini** (`google-genai`), **Pydantic v2 structured schemas**, and a **Local RAG Vector Store**.

## 🚀 Features

- **Multi-Track Role Preparation**: Practice across **DSA**, **System Design**, and **Behavioral** tracks.
- **RAG Vector Knowledge Base**: Built-in vector retrieval matching real-world questions with cosine similarity and metadata tagging.
- **Structured Question Generation**: Generates comprehensive interview prompts with constraints, progressive hints, and key milestones via Gemini.
- **5-Point Rubric Evaluation**: Rigorous evaluation across **Communication**, **Correctness**, and **Optimization** (scores 1-5), providing actionable strengths, areas for improvement, and ideal answer blueprints.
- **Flexible Setup**: Auto-detects `GEMINI_API_KEY` from `.env` or system environment, with sidebar input override and offline demo fallback.

## 🛠️ Quickstart

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd interview_prep_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   Create a `.env` file in the project directory:
   ```env
   GEMINI_API_KEY="your-gemini-api-key"
   ```

4. **Launch Application**:
   ```bash
   streamlit run app.py
   ```
