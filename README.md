# Intelligent_Complaint_Analysis

# Task-1 Exploratory Data Analysis and Data Preprocessing

This task focuses on performing exploratory data analysis (EDA) and preprocessing the CFPB consumer complaint dataset to prepare it for downstream NLP tasks such as embedding generation and RAG-based question answering.
  
## steps followed 

1. creating the Repository on github (https://github.com/nanecha/Intelligent_Complaint_Analysis.git)

2. git cloning  and  designing of standard folder structure 

3. Installing  required modules / see requirements.txt

4. writing and scripts on the folder of src and  notebooks  for visualization

## ✅ Summary of Activities

- Loaded the full CFPB consumer complaint dataset.
- Performed initial exploratory data analysis (EDA).
- Filtered dataset to focus on 5 selected financial products.
- Identified and visualized complaints with and without narratives.
- Cleaned the complaint narratives to remove noise and improve quality for embedding.
- Saved the cleaned dataset for future tasks.

📊 Exploratory Data Analysis (EDA)

-Checked data structure, types, and missing values.
- Visualized the distribution of complaints across products.
- Analyzed narrative length using histograms.
- Identified records with and without narrative text.

## 🔍 Data Filtering
Filtered the dataset to keep only complaints:

Belonging to the following products:

- Credit card
- Personal loan
- Buy Now, Pay Later (BNPL)
- Savings account
- Money transfers
- Containing non-empty consumer complaint narratives.

## 🧹 Narrative Cleaning
- Applied a text cleaning function to:
- Convert text to lowercase.
- Remove special characters, numbers, and boilerplate phrases.
- Eliminate extra whitespace.
A new column cleaned_narrative was added to the filtered dataset.

## 📚 Task 2: Text Chunking, Embedding, and Vector Store Indexing

This task processes complaint narratives by splitting them into smaller chunks, generating sentence embeddings using a pre-trained model, and storing them in a FAISS vector store for efficient semantic search.

## 🛠️ Setup & Requirements
Install the required dependencies:

-  pandas langchain sentence-transformers langchain-community faiss-cpu seaborn matplotlib scikit-learn
Optional 

## 📂 Directory Structure

Intelligent_Complaint_Analysis/
├── data/
│   └── filtered2_data.csv
├── faiss_store/
│   └── faiss_index
├── vector_store/
│   └── faiss_index/


## 🧾 Steps Performed
  1. 📥 Load Data

Load the preprocessed complaint data from the filtered2_data.csv file.

  2. 📏 Narrative Length Statistics

  3. ✂️ Text Chunking
Use RecursiveCharacterTextSplitter from LangChain to chunk long texts.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

Documents are chunked, and metadata such as complaint ID and product type are attached to each.

  4. 🔤 Embedding Generation

I Used sentence-transformers/all-MiniLM-L6-v2 to convert text into dense vectors.

   5. 🧠 Vector Store Creation (FAISS)

I Used LangChain’s FAISS wrapper to store the vectors along with their metadata.

   6. 📊 Visualization
I Use PCA to reduce the embedding vectors to 2D and plot using Seaborn.

# Task 3: Building the RAG Core Logic and Evaluation

## 📂 RAG Evaluation Pipeline

This module loads a pre-built FAISS vector store and embedding model to enable retrieval-augmented question answering over financial customer complaints.

## 🧱 Components

- FAISS Vector Store (pre-indexed)
- SentenceTransformer Embeddings (loaded via pickle)
- DistilGPT2 LLM (via transformers.pipeline)
- Prompt Template enforcing context-faithful answers
## ✅ Steps to Run:

Ensure the following files exist in your data directory:

- faiss_index (FAISS vector store)

- embedding_model.pkl (Serialized embedding model)

- Run rag_pipeline(question) for our input question.

Evaluate using the provided questions.

Results are stored in a DataFrame and can be saved to CSV.

## 💾 Output

evaluation_results.csv: Structured responses, retrieved source summaries, and placeholder quality scores.

# CrediTrust Complaint Analyzer (Task 4)

This project provides an interactive chatbot interface that allows users to query customer complaints using a Retrieval-Augmented Generation (RAG) pipeline built with FAISS, Sentence Transformers, and Hugging Face Transformers. The app is hosted with Gradio and is tailored for exploring customer complaint data from the CFPB dataset.

---

## 🔧 Installation

Before running the app, install all required dependencies:

pip install gradio pandas langchain langchain-community sentence-transformers faiss-cpu transformers torch


## 🚀 Running the Application
- On Google Colab:
- Upload the vector_store.zip:

-from google.colab import files
-uploaded = files.upload()
-!unzip vector_store.zip -d /content/
-Run the chatbot app using:
## 🤖 How It Works
 - Vector Store: Uses FAISS to store chunk embeddings of customer complaints.

-  Embedding Model: Uses SentenceTransformer to embed both questions and complaint texts.

- LLM: Uses google/flan-t5-base from Hugging Face Transformers for response generation.

- Interface: Gradio-based chat app to interact with the system.
