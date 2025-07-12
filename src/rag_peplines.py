import pandas as pd
from pathlib import Path
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.docstore.document import Document
import pickle

# Set up paths
VECTOR_STORE_PATH = Path('vector_store')
VECTOR_STORE_FILE = VECTOR_STORE_PATH / 'faiss_index'
EMBEDDING_MODEL_FILE = VECTOR_STORE_PATH / 'embedding_model.pkl'

# Load embedding model
print("Loading embedding model...")
with open(EMBEDDING_MODEL_FILE, 'rb') as f:
    embedding_model = pickle.load(f)

# Load vector store
print("Loading vector store...")
vector_store = FAISS.load_local(VECTOR_STORE_FILE, embeddings=embedding_model)

# Initialize LLM (using distilgpt2 for lightweight demo; replace with stronger model if available)
print("Loading language model...")
llm = pipeline('text-generation', model='distilgpt2', max_new_tokens=150, truncation=True)

# Prompt template
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based only on the provided context. If the context doesn't contain enough information to answer the question, state that clearly and do not make assumptions. Provide a concise and accurate answer.

Context:
{context}

Question:
{question}

Answer:
"""

def rag_pipeline(question, k=5):
    """
    RAG pipeline: Retrieve relevant chunks and generate an answer.
    
    Args:
        question (str): User's question
        k (int): Number of chunks to retrieve
    
    Returns:
        dict: Contains answer, retrieved documents, and their metadata
    """
    # Embed the question
    question_embedding = embedding_model.encode([question])[0]
    
    # Perform similarity search
    retrieved_docs = vector_store.similarity_search_by_vector(question_embedding, k=k)
    
    # Combine context from retrieved documents
    context = "\n".join([f"Complaint ID {doc.metadata['complaint_id']} (Product: {doc.metadata['product']}): {doc.page_content}" 
                         for doc in retrieved_docs])
    
    # Format prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Generate response
    response = llm(prompt, num_return_sequences=1)[0]['generated_text']
    
    # Extract answer (remove prompt from response)
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip() if "Answer:" in response else response.strip()
    
    return {
        'answer': answer,
        'retrieved_docs': retrieved_docs
    }

# Evaluation questions
eval_questions = [
    "What are common issues with credit card complaints?",
    "How do consumers describe problems with Buy Now, Pay Later services?",
    "Are there any complaints about unauthorized transactions in money transfers?",
    "What fees are mentioned in savings account complaints?",
    "What are typical delays reported in personal loan processing?"
]

# Run evaluation
evaluation_results = []
for question in eval_questions:
    result = rag_pipeline(question)
    # For simplicity, assume a manual quality score (1-5) based on relevance and coherence
    # In practice, you'd evaluate based on ground truth or human judgment
    evaluation_results.append({
        'question': question,
        'answer': result['answer'],
        'retrieved_sources': [
            {
                'complaint_id': doc.metadata['complaint_id'],
                'product': doc.metadata['product'],
                'text': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            } for doc in result['retrieved_docs'][:2]  # Show top 2 sources
        ],
        'quality_score': 3,  # Placeholder; adjust based on manual review
        'comments': "Placeholder: Evaluate coherence and relevance manually."
    })

# Save evaluation results as a CSV for reporting
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv('evaluation_results.csv', index=False)
print("Evaluation results saved to evaluation_results.csv")