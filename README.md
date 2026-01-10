# 📚 Semantic Book Recommender with LLMs

This repository contains the complete implementation of the **freeCodeCamp course**  
🎓 **Build a Semantic Book Recommender with LLMs – Full Course**

The project demonstrates how to build an **end-to-end semantic recommendation system** using **Large Language Models (LLMs)**, **vector databases**, **zero-shot classification**, **emotion-based sentiment analysis**, and a **Gradio web application**.

---

## 🚀 Project Overview

This system enables users to:
- Search for books using **natural language queries**
- Retrieve **semantically similar books** using vector embeddings
- Filter books by **fiction / non-fiction**
- Sort recommendations based on **emotional tone**
- Interact through a **web-based Gradio dashboard**

---


<img width="1919" height="894" alt="image" src="https://github.com/user-attachments/assets/8bfa708a-5846-44d0-b9a8-a431b2cb90b8" />
---

<img width="1919" height="924" alt="image" src="https://github.com/user-attachments/assets/9231aacc-e6c9-4f15-82ec-e313dffed57a" />
---

## 🧩 Project Components

### 1️⃣ Text Data Cleaning
📓 **Notebook:** `data-exploration.ipynb`

- Load and explore the Kaggle book dataset
- Clean titles, authors, and descriptions
- Handle missing values and inconsistencies
- Prepare text for embeddings and LLM processing

---

### 2️⃣ Semantic (Vector) Search
📓 **Notebook:** `vector-search.ipynb`

- Generate vector embeddings from book descriptions
- Store embeddings in a **Chroma vector database**
- Perform semantic similarity search
- Example query:
                "A book about a person seeking revenge"

  
---

### 3️⃣ Zero-Shot Text Classification
📓 **Notebook:** `text-classification.ipynb`

- Use LLMs for **zero-shot classification**
- Categorize books into:
- Fiction
- Non-Fiction
- Enables faceted filtering in the recommendation system

---

### 4️⃣ Sentiment & Emotion Analysis
📓 **Notebook:** `sentiment-analysis.ipynb`

- Uses a **Hugging Face emotion classification model**
- Extracts **7 emotional dimensions**:
- Joy
- Anger
- Sorrow
- Fear
- Surprise
- Disgust
- Neutral
- Allows sorting books by emotional tone
(joyful, suspenseful, sad, etc.)

---

### 5️⃣ Web Application (Gradio)
🧠 **File:** `gradio-dashboard.py`

- Interactive recommendation interface built with **Gradio**
- Features:
- Natural language book search
- Fiction / Non-Fiction filtering
- Emotion-based ranking
- Fully integrated with vector search and LLM pipelines

---

## 🛠️ Tech Stack

- Python 3.11
- OpenAI API
- Hugging Face Transformers
- LangChain
- Chroma Vector Database
- Gradio
- Pandas, Matplotlib, Seaborn

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`.

Key libraries:

    kagglehub
    pandas
    matplotlib
    seaborn
    python-dotenv
    langchain-community
    langchain-opencv
    langchain-chroma
    transformers
    gradio
    notebook
    ipywidgets



---

## 🔐 Environment Variables

Create a `.env` file in the **root directory** with the following content:

```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

⚠️ Do NOT commit the .env file to GitHub
Ensure .env is included in .gitignore.

##📂 Dataset

Dataset is sourced from Kaggle
Downloaded using kagglehub
Instructions for dataset setup are included in the notebooks

## ▶️ How to Run the Project
### 1️⃣ Clone the Repository

    git clone https://github.com/your-username/semantic-book-recommender.git
    cd semantic-book-recommender
### 2️⃣ Install Dependencies

    pip install -r requirements.txt

### 3️⃣ Set Up Environment Variables

    Create the .env file with your OpenAI and Hugging Face credentials.

### 4️⃣ Run Notebooks (in order)

    data-exploration.ipynb
    vector-search.ipynb
    text-classification.ipynb
    sentiment-analysis.ipynb

### 5️⃣ Launch the Web App

    python gradio-dashboard.py

## 🌟 Key Features

🔍 Semantic search using vector embeddings
🧠 Zero-shot classification with LLMs
😊 Emotion-aware recommendations (7 emotions)
⚡ Fast vector similarity search with Chroma
🖥️ Clean and interactive Gradio interface

## 📘 Course Credit

This project follows the freeCodeCamp tutorial:
Build a Semantic Book Recommender with LLMs – Full Course
All educational credit belongs to the original course creators.

📜 License

This project is intended for educational and learning purposes.
Please review dataset and model licenses before commercial use.

🤝 Contributions

Contributions, issues, and pull requests are welcome.

⭐ If you found this project useful, consider starring the repository!
