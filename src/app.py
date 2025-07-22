import gradio as gr
import pandas as pd
from pathlib import Path
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pickle


def run_rag_pipeline( VECTOR_STORE_FILE, question=None, k=5, 
                     llm_model='google/flan-t5-base'):
    """
    Run the RAG pipeline: retrieve relevant complaint 
    chunks and generate an answer.
    
    Args:
        vector_store_path (str): Directory containing the FAISS vector store and embedding model
        question (str): User question to process
        k (int): Number of chunks to retrieve
        llm_model (str): Hugging Face model for text generation (default: google/flan-t5-base)

    Returns:
        dict: Contains answer and retrieved documents
    """
    # Set up paths
    VECTOR_STORE_FILE = 'F:/Intelligent_Complaint_Analysis/data/faiss_index'
    EMBEDDING_MODEL_FILE = 'F:/Intelligent_Complaint_Analysis/data/embedding_model.pkl'

    # Load embedding model
    print("Loading embedding model...")
    with open(EMBEDDING_MODEL_FILE, 'rb') as f:
        embedding_model = pickle.load(f)

    # Load vector store
    print("Loading vector store...")
    vector_store = FAISS.load_local(VECTOR_STORE_FILE, embeddings=embedding_model, allow_dangerous_deserialization=True)

    # Initialize LLM
    print(f"Loading language model ({llm_model})...")
    llm = pipeline('text2text-generation', model=llm_model, max_length=150)

    # Prompt template
    PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based only on the provided context. If the context doesn't contain enough information to answer the question, state that clearly and do not make assumptions. Provide a concise and accurate answer.

Context:
{context}

Question:
{question}

Answer:
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
    
    # Extract answer
    answer = response.strip()
    
    return {
        'answer': answer,
        'retrieved_docs': retrieved_docs
    }


def chat_interface(question, history):
    """
    Gradio chat interface function to handle user questions 
    and display answers with sources.
    
    Args:
        question (str): User's question
        history (list): Chat history
    
    Returns:
        tuple: (answer, sources, updated history)
    """
    if not question:
        return "Please enter a question.", "", history
    
    # Run RAG pipeline
    result = run_rag_pipeline(question=question)
    
    # Format answer
    answer = result['answer']
    
    # Format sources
    sources = "\n\n".join([f"**Source {i+1} (Complaint ID {doc.metadata['complaint_id']}, Product: {doc.metadata['product']})**:\n{doc.page_content[:200]}..." 
                          for i, doc in enumerate(result['retrieved_docs'][:3])])
    
    # Update chat history
    history.append((question, f"{answer}\n\n**Retrieved Sources**:\n{sources}"))
    
    return answer, sources, history


def clear_conversation():
    """
    Clear the chat history.
    
    Returns:
        tuple: Empty history and cleared outputs
    """
    return [], "", ""

# Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analyzer") as demo:
    gr.Markdown("# CrediTrust Complaint Analyzer")
    gr.Markdown("Ask questions about customer complaints from the CFPB dataset.")
    
    # Chatbot for conversation history
    chatbot = gr.Chatbot(label="Conversation")
    
    # Question input
    question = gr.Textbox(label="Your Question", placeholder="e.g., What are common issues with credit card complaints?")
    
    # Buttons
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear")
    
    # Output for answer
    answer_output = gr.Textbox(label="Answer", interactive=False)
    
    # Output for sources
    sources_output = gr.Textbox(label="Retrieved Sources", interactive=False)
    
    # Bind submit button
    submit_button.click(
        fn=chat_interface,
        inputs=[question, chatbot],
        outputs=[answer_output, sources_output, chatbot]
    )
    
    # Bind clear button
    clear_button.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, answer_output, sources_output]
    )

# Launch the app
if __name__ == "__main__":
    pass

    #demo.launch()