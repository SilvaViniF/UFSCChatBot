import pymupdf4llm
import pymupdf
import bs4 as BeautifulSoup
import os
import pandas as pd
from chonkie import SemanticChunker, SemanticChunk

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

def _process_file(file_path: str, max_length: int) -> list[str]:
    
    chunker = SemanticChunker(chunk_size=max_length)
    
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
    
    semantic_chunks = chunker.chunk(text)
    chunks = []
    for chunk in semantic_chunks:
        chunks.append(chunk.text)
    return chunks
    
#TODO add cache

def get_documents(folder_path: str, max_length: int = 512) -> list[str]:
    documents = []
    for file_path in os.listdir(folder_path):
        documents.append(_process_file(f"{folder_path}/{file_path}",max_length))
    return documents
#endregion