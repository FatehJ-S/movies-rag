import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import openai

# Set your OpenAI API key here or use Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load model, index, and data only once
@st.cache_resource
def load_resources():
    df = pd.read_csv("clean_movies.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("movie_index.faiss")
    return df, model, index

df, model, index = load_resources()

# Search function using FAISS
def search_movies(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = normalize(np.array(query_embedding).astype("float32"), axis=1)
    distances, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]]
    return results

# Generate answer using GPT model
def generate_answer(query, top_k=5):
    results = search_movies(query, top_k)
    context = "\n\n".join(results['description'].fillna("").tolist())

    prompt = f"Answer the question based on the movie data below:\n\n{context}\n\nQuestion: {query}"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, results

# Streamlit app UI
st.title("Movie Q&A with RAG")
query = st.text_input("Ask a question about movies:", placeholder="For example: Which movies involve time travel?")

top_k = st.slider("Number of retrieved documents", 1, 10, 5)

if st.button("Ask") and query:
    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(query, top_k)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Movie Descriptions")
        for _, row in sources.iterrows():
            st.markdown(f"**{row.get('title', 'Unknown Title')}**")
            st.write(row.get("description", "No description available"))
