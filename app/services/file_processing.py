import pymupdf
import bs4 as BeautifulSoup
import os

def process_html(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, "html.parser")
    return soup.get_text(separator="\n").strip()

def process_pdf(file_path: str) -> str:
    doc = pymupdf.open(file_path)
    return "\n".join(page.get_text() for page in doc)

def process_file(file_path: str) -> str:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".html":
        return process_html(file_path)
    elif extension == ".pdf":
        return process_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")





text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # Maximum size of each chunk
    chunk_overlap=50  # Number of overlapping characters between chunks
)

def split_long_text(text: str, max_length: int = 1024) -> List[str]:
    # Use the text splitter to split the text
    return text_splitter.split_text(text)

def process_file(file_path: str, max_length: int) -> List[str]:
    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.html':
            return process_html(file_path, max_length)
        elif file_extension == '.pdf':
            return process_pdf(file_path, max_length)
        elif file_extension == '.csv':
            return process_csv(file_path, max_length)
        elif file_extension in ['.txt', '.py']:  # Add support for .txt and .py files
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return split_long_text(content, max_length)
        else:
            print(f"Unsupported file type: {file_path}")
            return []
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def process_html(file_path: str, max_length: int) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, "html.parser")
        content = soup.get_text(separator="\n").strip()
        return split_long_text(content, max_length)

def process_pdf(file_path: str, max_length: int) -> List[str]:
    try:
        doc = fitz.open(file_path)
        content = ""
        for page in doc:
            content += page.get_text()
        return split_long_text(content.strip(), max_length)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def process_csv(file_path: str, max_length: int) -> List[str]:
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        content = df.to_csv(index=False)
        return split_long_text(content.strip(), max_length)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def get_documents(file_path: List[str], max_length: int = 512, cache_file: str = 'documents_cache.pkl') -> List[str]:
    # Get the directory of the current file (llama3.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert cache_file to an absolute path within the backend directory
    cache_file = os.path.join(current_dir, cache_file)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for cache file: {cache_file}")
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        print(f"Loading documents from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Cache file not found. Processing documents...")
    with multiprocessing.Pool() as pool:
        documents = pool.map(partial(process_file, max_length=max_length), file_path)
    documents = [doc for file_docs in documents for doc in file_docs]
    print(f"Total document chunks loaded: {len(documents)}")

    # Save the processed documents to the cache file
    print(f"Saving documents to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(documents, f)

    return documents
#endregion