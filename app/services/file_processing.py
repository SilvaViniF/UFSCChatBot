import pymupdf4llm
import pymupdf
import bs4 as BeautifulSoup
import os
import pandas as pd
from chonkie import SDPMChunker, SemanticChunk

def _process_pdf(file_path: str) -> list[str]:
    try:
        doc = pymupdf.open(file_path)
        markdown = pymupdf4llm.to_markdown(doc)
        return markdown
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def _process_html(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, "html.parser")
        content = soup.get_text(separator="\n").strip()
        return content
    
def _process_csv(file_path: str) -> list[str]:
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        content = df.to_csv(index=False)
        return content
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def _chunk_file(text: str, max_length: int) -> list[SemanticChunk]:
    chunker = SDPMChunker(
    embedding_model="minishlab/potion-base-8M",  # Default model
    similarity_threshold=0.5,                   # Similarity threshold (0-1)
    chunk_size=max_length,                             # Maximum tokens per chunk
    initial_sentences=1,                        # Initial sentences per chunk
    skip_window=1                               # Number of chunks to skip when looking for similarities
    )
    return chunker.chunk(text)

def process_file(file_path: str, max_length: int) -> list[str]:
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.html':
            text = _process_html(file_path)
        elif file_extension == '.pdf':
            text = _process_pdf(file_path)
        elif file_extension == '.csv':
            text = _process_csv(file_path)
        elif file_extension in ['.txt']:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            print(f"Unsupported file type: {file_path}")
            return []
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []
    
    semantic_chunks = _chunk_file(text,max_length)
    chunks = []
    for chunk in semantic_chunks:
        chunks.append(chunk.text)
    return chunks
    
#TODO add cache






def get_documents(file_path: list[str], max_length: int = 512, cache_file: str = 'documents_cache.pkl') -> list[str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(current_dir, cache_file)
    
    documents = process_file(file_path,max_length)
    documents = [doc for file_docs in documents for doc in file_docs]
    print(f"Total document chunks loaded: {len(documents)}")

    # Save the processed documents to the cache file
    print(f"Saving documents to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(documents, f)

    return documents
#endregion