from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate  # Add this import
import numpy as np  # Add this import if not present
import params
import uvicorn
import warnings
import os

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI(
    title="FixConnect Chatbot API",
    description="API for the FixConnect customer service chatbot",
    version="1.0.0"
)

# Initialize OpenAI client
llm = OpenAI(api_key=params.openai_api_key, temperature=0.7)

# Initialize MongoDB and vector store
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)
vector_store = MongoDBAtlasVectorSearch(
    collection, 
    embeddings, 
    index_name=params.index_name
)

# Basic conversation patterns
BASIC_CONVERSATIONS = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! How may I assist you?",
    "hey": "Hey! What can I do for you?",
    "bye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Let me know if you need anything else.",
    "thank you": "You're welcome! Is there anything else I can help you with?",
    "how are you": "I'm doing well, thank you! How can I assist you today?",
    "good morning": "Good morning! How may I help you?",
    "good afternoon": "Good afternoon! What can I do for you?",
    "good evening": "Good evening! How can I assist you?"
}

class Query(BaseModel):
    question: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Welcome to FixConnect Chatbot API"}

def calculate_similarity_score(question: str, faq_question: str) -> float:
    """Calculate similarity score between user question and FAQ question"""
    try:
        # Get embeddings for both questions
        q1_embedding = embeddings.embed_query(question)
        q2_embedding = embeddings.embed_query(faq_question)
        
        # Calculate cosine similarity
        similarity = np.dot(q1_embedding, q2_embedding) / (
            np.linalg.norm(q1_embedding) * np.linalg.norm(q2_embedding)
        )
        return float(similarity)
    except Exception:
        return 0.0

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    try:
        question = query.question.lower().strip()
        
        # Handle basic conversations
        if question in BASIC_CONVERSATIONS:
            return {"answer": BASIC_CONVERSATIONS[question]}
        
        # Get similar FAQ entries
        docs = vector_store.similarity_search_with_score(question, k=1)
        
        if docs:
            doc, score = docs[0]
            try:
                content = json.loads(doc.page_content)
                faq_question = content.get("question", "")
                similarity_score = calculate_similarity_score(question, faq_question)
                
                # If similarity is high (>0.9), return FAQ answer directly
                if similarity_score > 0.9:
                    return {"answer": content.get("answer", doc.page_content.strip())}
                    
                # If similarity is lower, generate contextual response using LLM
                else:
                    template = """
                    Based on the user's question and the closest FAQ entry, provide a helpful and specific response.
                    User Question: {question}
                    Related FAQ: {faq_content}
                    Generate a natural and contextual response focusing on the user's specific needs.
                    If the user asks about a specific service, recommend that service and explain why.
                    """
                    
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["question", "faq_content"]
                    )
                    
                    response = llm(prompt.format(
                        question=question,
                        faq_content=doc.page_content
                    ))
                    
                    return {"answer": response.strip()}
                    
            except json.JSONDecodeError:
                return {"answer": doc.page_content.strip()}
        
        return {"answer": "I'm sorry, I couldn't find relevant information for that query."}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  # Changed from main:app to app:app
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )