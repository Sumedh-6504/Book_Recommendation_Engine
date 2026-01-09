from dotenv import load_dotenv
import pandas as pd
import numpy as np
import time

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import gradio as gr

load_dotenv()

# Load dataset
books = pd.read_csv("Book_Recommendation_Engine/data/books_with_emotions.csv")
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["thumbnail"] + "&fife=w800"
)

# Load raw text file
raw_documents = TextLoader("Book_Recommendation_Engine/tagged_description.txt", encoding='utf8').load()

# Split text line by line
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
    separators=["\n"]
)
documents = text_splitter.split_documents(raw_documents)

# ----------- BATCHED EMBEDDING (NO RATE LIMIT ERRORS) -------------
embedding = OpenAIEmbeddings()
db_books = Chroma(collection_name="books", embedding_function=embedding)

batch_size = 150  # SAFE → produces <40k tokens/min

for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i + batch_size]
    db_books.add_documents(batch_docs)
    print(f"Indexed batch {i // batch_size + 1}")
    time.sleep(2)  # adds safety buffer for TPM limits


# ------------------------------------------------------------------------


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    # extract isbn numbers from line format "9781xxxxx description"
    books_list = []
    for r in recs:
        first_token = r.page_content.strip('"').split()[0]
        if first_token.isdigit():
            books_list.append(int(first_token))

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Filter category
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    else:
        book_recs = book_recs.head(final_top_k)


    # Emotion tone
    if tone == "Happy":
        book_recs.sort_values("joy", ascending=False, inplace = True)
    elif tone == "Surprising":
        book_recs.sort_values("surprise", ascending=False, inplace = True)
    elif tone == "Angry":
        book_recs.sort_values("anger", ascending=False, inplace = True)
    elif tone == "Suspenseful":
        book_recs.sort_values("fear", ascending=False, inplace = True)
    elif tone == "Sad":
        book_recs.sort_values("sadness", ascending=False, inplace = True)

    return book_recs


def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        desc = row["description"].split()
        truncated_description = " ".join(desc[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            authors_str = authors[0]

        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Describe a book you want:")
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotion Tone", value="All")
        submit_button = gr.Button("Find Books")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(columns=8, rows=2)

    submit_button.click(recommend_books, [user_query, category_dropdown, tone_dropdown], output)

if __name__ == "__main__":
    dashboard.launch()

