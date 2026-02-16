from django.shortcuts import render
from .models import CompanyInfo
from sentence_transformers import SentenceTransformer
from pgvector.django import L2Distance
from transformers import pipeline

# --- Load Models (Global to avoid reloading on every request) ---

# 1. Retriever Model (Finds the right paragraph)
# Converts text to vectors
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Reader Model (Extracts the specific answer)
# 'distilbert-base-cased-distilled-squad' is a free, fast model trained to answer questions
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def search_view(request):
    answer = ""
    query = request.POST.get('query', '')
    
    if request.method == 'POST' and query:
        # Step 1: Vector Search (Find the right Company)
        query_vector = embedder.encode(query)
        
        # Find the single most similar record in the database
        best_match = CompanyInfo.objects.order_by(
            L2Distance('embedding', query_vector)
        ).first()
        
        if best_match:
            # Step 2: AI Extraction (Find the Answer in the text)
            # We tell the AI: "Here is the text about a company. Based on this text, answer the user's question."
            
            ai_response = qa_model(
                question=query,
                context=best_match.content
            )
            
            # The pipeline returns a dictionary: {'score': 0.9, 'start': 10, 'end': 15, 'answer': 'Technology'}
            # We just want the 'answer' text.
            answer = ai_response['answer']
            
            # Optional: If you want to see which company it found for debugging, you can append it:
            # answer = f"{ai_response['answer']} (Source: {best_match.name})"
        else:
            answer = "I couldn't find any relevant company in the database."

    return render(request, 'index.html', {'answer': answer, 'query': query}) 