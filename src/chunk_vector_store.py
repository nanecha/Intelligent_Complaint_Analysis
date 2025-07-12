import pandas as pd
from langchain.docstore.document import Document
import uuid


def chunk_cleaned_complaints_to_documents(df, text_splitter, batch_size=1000):
    """
    Process a DatacleqFrame of complaints to create chunked Document objects with metadata.
    Args:
        df (pandas.DataFrame): DataFrame containing complaint data with 'cleaned_narrative', 
                              'product', and optionally 'Complaint ID' columns.
        text_splitter (langchain.text_splitter.RecursiveCharacterTextSplitter):
        Text splitter for chunking narratives.batch_size (int): Number of rows to process per batch.

    Returns:
        list: List of langchain Document objects with chunked narratives and metadata.
    """
    documents = []
    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        for idx, row in batch_df.iterrows():
            try:
                complaint_id = str(row.get('Complaint ID', uuid.uuid4()))
                product = row.get('product', 'Unknown')
                narrative = row['cleaned_narrative']
                if pd.isna(narrative) or not narrative.strip():
                    print(
                        f"Skipping empty narrative for Complaint ID {complaint_id}")
                    continue
                chunks = text_splitter.split_text(narrative)
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'complaint_id': complaint_id,
                            'product': product,
                            'chunk_index': chunk_idx
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        print(
            f"Processed batch {start_idx // batch_size + 1}: {len(documents)} chunks so far")
    return documents


# Example usage
if __name__ == "__main__":
    pass
