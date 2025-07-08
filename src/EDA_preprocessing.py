import pandas as pd
import re


def load_and_process_complaints(file_path, chunksize=100000):
    """
    Loads a large complaints CSV file in chunks, cleans columns, 
    and extracts relevant data.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file (e.g., 
        'F:/Intelligent_Complaint_Analysis/data/Complaints.csv')
    chunksize : int
        Number of rows to read per chunk (default is 100,000)

    Returns:
    -------
    df : pd.DataFrame
        Combined cleaned DataFrame with selected columns.
    summary : dict
        Dictionary containing processing summary: 
        total rows, with and without narrative.
    """
    total_raw = 0
    total_with_narrative = 0
    total_without_narrative = 0
    df = pd.DataFrame()

    for i, chunk in enumerate(pd.read_csv(file_path,
                                          chunksize=chunksize,
                                          low_memory=False)):
        print(f'🔄 Processing chunk {i+1}...')

        # Clean column names
    chunk.columns = chunk.columns.str.strip()

        # Update total row count
    total_raw += len(chunk)

        # Select and rename relevant columns
    chunk = chunk[['Complaint ID', 'Product',
                       'Consumer complaint narrative']].copy()

    chunk.columns = ['complaint_id', 'product', 'narrative']

        # Update narrative counts
    total_with_narrative += chunk['narrative'].notna().sum()
    total_without_narrative += chunk['narrative'].isna().sum()

        # Append cleaned chunk
    df = pd.concat([df, chunk], ignore_index=True)

    # Return the final DataFrame and summary info
    summary = {
        "total_raw": total_raw,
        "with_narrative": total_with_narrative,
        "without_narrative": total_without_narrative,
        "final_shape": df.shape
    }

    return df, summary

# Text cleaning function


def clean_narrative(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove boilerplate phrases
    boilerplate = [
        r'i am writing to file a complaint',
        r'please assist',
        r'thank you for your attention',
        r'xx+',  # Remove masked data (e.g., XXXX)
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
