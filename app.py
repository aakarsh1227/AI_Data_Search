import streamlit as st
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Configuration ---
DB_CONFIG = {
    "dbname": "company_vectors",
    "user": "db_owners",
    "password": "secure_pass_123",
    "host": "localhost",
    "port": "5432"
}

# --- 1. Load AI Models ---
@st.cache_resource
def load_models():
    # Retriever: Finds the right paragraph
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Reader: Extracts the EXACT answer (We use RoBERTa for better accuracy)
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embedder, qa_model

embedder, qa_model = load_models()

# --- 2. Database Functions ---
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn

def reset_and_load_data():
    """Reloads data with Natural Language formatting for better AI answers."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create Table
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS streamlit_companies (
            id SERIAL PRIMARY KEY,
            name TEXT,
            content TEXT,
            embedding vector(384)
        );
    """)
    
    # Clear existing data so we don't duplicate
    cur.execute("TRUNCATE TABLE streamlit_companies;")
    
    try:
        df = pd.read_csv('data.csv')
        st.info(f"Processing {len(df)} rows... converting to natural language.")
        
        for _, row in df.iterrows():
            # KEY CHANGE: Format as Natural English Sentences
            # This helps the AI extract "Apple" or "387 Billion" specifically.
            text_repr = (
                f"The company name is {row['Company Name']}. "
                f"It operates in the {row['Industry']} industry, specifically the {row['Sector']} sector. "
                f"The headquarters is located in {row['HQ State']}. "
                f"The annual revenue is ${row['Annual Revenue 2022-2023 (USD in Billions)']} Billion. "
                f"The company has {row['Employee Size']} employees."
            )
            
            # Generate Vector
            vector = embedder.encode(text_repr)
            
            # Insert into DB
            cur.execute(
                "INSERT INTO streamlit_companies (name, content, embedding) VALUES (%s, %s, %s)",
                (row['Company Name'], text_repr, vector)
            )
        
        conn.commit()
        st.success("Database updated! Now the AI can read the data better.")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        conn.close()

def search_and_answer(query):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Find the best matching company
    query_vector = embedder.encode(query)
    cur.execute("""
        SELECT content FROM streamlit_companies 
        ORDER BY embedding <-> %s::vector 
        LIMIT 1;
    """, (query_vector,))
    
    result = cur.fetchone()
    conn.close()
    
    if result:
        context_text = result[0]
        # 2. Extract the EXACT answer from the text
        ai_response = qa_model(question=query, context=context_text)
        return ai_response['answer'], context_text
    else:
        return None, None

# --- 3. The Streamlit UI ---
st.title("AI Company Search Engine")

# Sidebar to Reset Data
with st.sidebar:
    st.header("System")
    if st.button("Reset & Reload Database"):
        reset_and_load_data()
    st.info("Click this once to update the data format!")

# Chat Interface
query = st.text_input("Ask a specific question:")

if st.button("Ask") and query:
    with st.spinner("Analyzing..."):
        answer, context = search_and_answer(query)
        
        if answer:
            # Display the EXACT answer nicely
            st.markdown("### **Answer:**")
            st.success(answer)
            
            # Hide the full text in an expander
            with st.expander("Show Evidence (Source Text)"):
                st.write(context)
        else:
            st.error("I couldn't find an answer to that.")