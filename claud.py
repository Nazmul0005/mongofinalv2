from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate
import numpy as np
import params
import uvicorn
import warnings
import os
import re

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize FastAPI app
app = FastAPI(
    title="FixConnect Chatbot API",
    description="API for the FixConnect customer service chatbot",
    version="1.0.0"
)

# Initialize OpenAI client with specific parameters for shorter responses
llm = OpenAI(
    api_key=params.openai_api_key, 
    temperature=0.3,  # Lower temperature for more consistent, concise responses
    max_tokens=150,   # Limit response length
    model_name="gpt-3.5-turbo-instruct"  # Faster model for shorter responses
)

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
    "hi": "Hello! How can I help you with FixConnect today?",
    "hello": "Hi there! How may I assist you with FixConnect?",
    "hey": "Hey! What can I do for you regarding FixConnect?",
    "bye": "Goodbye! Have a great day using FixConnect!",
    "thanks": "You're welcome! Let me know if you need anything else about FixConnect.",
    "thank you": "You're welcome! Is there anything else I can help you with regarding FixConnect?",
    "how are you": "I'm doing well, thank you! How can I assist you with FixConnect today?",
    "good morning": "Good morning! How may I help you with FixConnect?",
    "good afternoon": "Good afternoon! What can I do for you regarding FixConnect?",
    "good evening": "Good evening! How can I assist you with FixConnect?"
}

# FixConnect Knowledge Base
FIXCONNECT_KNOWLEDGE = """
FixConnect App Knowledge Base:

ABOUT FIXCONNECT:
- FixConnect is a mobile application that connects customers with certified engineers for technical services
- Main purpose: "Find Trusted Engineers Fast. Easily connect with certified professionals for any technical issue — anytime, anywhere"
- Serves as a platform for customers to find and hire engineers for various technical services
- Emphasizes trust and certification of professionals

AVAILABLE SERVICES:
- Electrical Services (circuit issues, wiring, electrical repairs)
- HVAC Systems (heating, ventilation, air conditioning)
- Security Systems (installation, maintenance, monitoring)
- Power Solutions (power issues, backup systems)
- Plumbing (leak repairs, pipe issues, water systems)

USER ROLES:
- Customer: Users who need technical services
- Engineer: Certified professionals who provide technical services

REGISTRATION PROCESS:
For Customers:
1. Tap Register → Choose 'Customer' role
2. Enter username, email, password, location
3. Verify email with code sent to phone
4. Continue to access the app

For Engineers:
1. Tap Register → Choose 'Engineers' role
2. Enter username, email, password, location
3. Upload Trade Certification
4. Set availability and write description
5. Verify email with code
6. Continue to access the app

SERVICE REQUEST PROCESS:
1. Customer taps "Request a Services"
2. Fill out quote form: Service Name, Location, Mobile Number, Service Date, Description
3. Optional: Upload image of the issue
4. Tap "Send Request"
5. Engineers see request in their Services Request List
6. Engineers view details and send offers with price and time
7. Customer receives Offer List from interested engineers
8. Customer can Accept or Decline offers
9. After acceptance, proceed to payment and booking confirmation

PAYMENT AND BOOKING:
- Payment method: Stripe (secure payment processing)
- Payment process: Review details → Confirm Stripe payment → Pay Now → Confirm Booking → Done
- All major credit/debit cards accepted through Stripe
- Option for extra payment requests for additional services

COMMUNICATION:
- Built-in Messages/Chat feature for direct communication
- Available after booking confirmation
- Allows coordination between customers and engineers

SUBSCRIPTION PLANS (for Engineers):
- Free Plan: £0 - "Jump In & Explore"
- Pro Plan: £20/month - "Scaling Your Practice" 
- Premium Plan: £49/month - "More Leads, Bigger Profits"
- Payment via Stripe

PROFILE MANAGEMENT:
- User Information (edit details, update trade certification for engineers)
- Subscription management
- Security settings
- Change Password option
- Legal and Policies access
- Log Out functionality

RATING AND REVIEW:
- Customers can rate and review engineers after service completion
- Helps build trust and inform future customer decisions
- Rating system for service quality

NOTIFICATIONS:
- New Job Request Available (for engineers)
- Job Accepted - Prepare for Visit
- Payment Released for Completed Job
- Message notifications

POLICIES:
Terms of Service:
- Must be 18+ to use the app
- Follow all applicable laws and community standards
- Account suspension/termination for misuse or fraudulent activities
- Platform not responsible for financial outcomes
- Engineers are independent and responsible for their content

Privacy Policy:
- Collects: name, email, payment info, usage data, communication preferences
- Uses data to: provide/improve app, process transactions, personalize experience
- Never sells information to third parties

Refund Policy:
- All purchases are final
- No refunds due to digital/time-sensitive nature
- Contact support for technical issues

TECHNICAL FEATURES:
- Email verification during registration
- Password reset functionality
- Secure payment processing
- Real-time messaging
- Image upload capability
- Location-based services
- Rating and review system
"""

# Topic validation keywords
FIXCONNECT_KEYWORDS = [
    'fixconnect', 'engineer', 'engineers', 'service', 'services', 'electrical', 'hvac', 
    'plumbing', 'security', 'power', 'quote', 'booking', 'payment', 'stripe', 'register',
    'registration', 'login', 'password', 'reset', 'customer', 'offer', 'chat', 'message',
    'rating', 'review', 'subscription', 'plan', 'certification', 'technical', 'repair',
    'maintenance', 'installation', 'issue', 'problem', 'fix', 'connect', 'app', 'mobile',
    'profile', 'account', 'verification', 'code', 'policy', 'terms', 'refund', 'privacy',
    'support', 'help', 'assistance', 'professional', 'certified', 'trusted', 'domestic',
    'home', 'house', 'circuit', 'wiring', 'leak', 'pipe', 'heating', 'cooling', 'air',
    'conditioning', 'ventilation', 'security system', 'power solution', 'logout', 'log out'
]

def is_fixconnect_related(question: str) -> bool:
    """Check if the question is related to FixConnect app"""
    question_lower = question.lower()
    
    # Check for FixConnect keywords
    for keyword in FIXCONNECT_KEYWORDS:
        if keyword in question_lower:
            return True
    
    # Check for common service-related patterns
    service_patterns = [
        r'\b(need|want|looking for|require).*(help|service|engineer|technician|professional)\b',
        r'\b(electrical|plumbing|hvac|security|power).*(issue|problem|repair|fix|service)\b',
        r'\b(how to|how do i|can i).*(register|login|book|pay|contact|message)\b',
        r'\b(my|i have|there is).*(issue|problem|trouble).*(electrical|plumbing|hvac|power|circuit|leak)\b'
    ]
    
    for pattern in service_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False

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
        question = query.question.strip()
        question_lower = question.lower()
        
        # Handle basic conversations
        if question_lower in BASIC_CONVERSATIONS:
            return {"answer": BASIC_CONVERSATIONS[question_lower]}
        
        # Check if question is related to FixConnect
        if not is_fixconnect_related(question):
            return {"answer": "I'm sorry, I can only help you with questions related to FixConnect app and its services. Please ask me about FixConnect features, services, registration, booking, or any technical issues you need help with."}
        
        # Get similar FAQ entries
        docs = vector_store.similarity_search_with_score(question, k=1)
        
        if docs:
            doc, score = docs[0]
            
            # Debug: Print what we're getting from MongoDB
            print(f"Debug - Raw doc content: {doc.page_content}")
            print(f"Debug - Vector search score: {score}")
            
            try:
                content = json.loads(doc.page_content)
                faq_question = content.get("question", "")
                faq_answer = content.get("answer", "")
                
                print(f"Debug - FAQ Question: {faq_question}")
                print(f"Debug - FAQ Answer: {faq_answer}")
                
                # Calculate similarity between user question and FAQ question
                similarity_score = calculate_similarity_score(question, faq_question)
                print(f"Debug - Similarity score: {similarity_score}")
                
                # If similarity is high OR vector search score is high, return FAQ answer EXACTLY
                if similarity_score > 0.7 or score > 0.8:
                    print(f"Debug - Returning direct FAQ answer")
                    return {"answer": faq_answer}
                else:
                    print(f"Debug - Using LLM for contextual response")
                    # Generate SHORT contextual response using LLM
                    template = """
                    You are a FixConnect customer service assistant. Provide SHORT, CONCISE answers using ONLY the FixConnect knowledge base.
                    
                    FixConnect Knowledge Base:
                    {knowledge_base}
                    
                    User Question: {question}
                    Related FAQ: {faq_content}
                    
                    Instructions:
                    1. Keep response SHORT (maximum 2-3 sentences)
                    2. Use ONLY FixConnect knowledge base information
                    3. Be direct and specific
                    4. For service issues, briefly mention the relevant FixConnect service
                    5. No generic advice - only FixConnect-specific information
                    
                    Response:
                    """
                    
                    prompt = PromptTemplate(
                        template=template,
                        input_variables=["question", "faq_content", "knowledge_base"]
                    )
                    
                    response = llm(prompt.format(
                        question=question,
                        faq_content=doc.page_content,
                        knowledge_base=FIXCONNECT_KNOWLEDGE
                    ))
                    
                    return {"answer": response.strip()}
                    
            except json.JSONDecodeError:
                print(f"Debug - JSON decode error, returning raw content")
                # If FAQ is not in JSON format, return it directly
                return {"answer": doc.page_content.strip()}
        
        # If no docs found, use knowledge base to provide SHORT relevant FixConnect information
        print(f"Debug - No docs found, using knowledge base")
        template = """
        You are a FixConnect customer service assistant. Provide SHORT, CONCISE answers using ONLY the FixConnect knowledge base.
        
        FixConnect Knowledge Base:
        {knowledge_base}
        
        User Question: {question}
        
        Instructions:
        1. Keep response SHORT (maximum 1 sentences)
        2. Use ONLY FixConnect knowledge base information
        3. Be direct and specific
        4. For service issues, briefly mention how to use FixConnect
        
        Response:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "knowledge_base"]
        )
        
        response = llm(prompt.format(
            question=question,
            knowledge_base=FIXCONNECT_KNOWLEDGE
        ))
        
        return {"answer": response.strip()}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred while processing your request: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )