import os
import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

folder_path = "RAG_test"
token = config.token
#not being used now
def extract_info_from_html(file_path, index):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        
        # Extract URL, Title, and content from the HTML
        url = soup.find("meta", property="og:url")["content"] if soup.find("meta", property="og:url") else "URL not found"
        title = soup.title.string if soup.title else "Title not found"
        content = soup.get_text(separator="\n").strip()
        
        return {"ID": index, "url": url, "Title": title, "content": content}


def extract_info_from_pdf(file_path, index):
    try:
        doc = fitz.open(file_path)
        content = ""
        for page in doc:
            content += page.get_text()
        title = os.path.splitext(os.path.basename(file_path))[0]
        
        return {"ID": index, "url": "URL not applicable", "Title": title, "content": content.strip()}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"ID": index, "url": "URL not found", "Title": "Title not applicable", "content": ""}


def extract_info_from_csv(file_path, index):
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        content = df.to_csv(index=False)
        title = os.path.splitext(os.path.basename(file_path))[0]
        
        return {"ID": index, "url": "URL not applicable", "Title": title, "content": content.strip()}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"ID": index, "url": "URL not found", "Title": "Title not applicable", "content": ""}


def split_into_chunks(content, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(content)
    return chunks

data = []
chunk_id = 1
for index, file_name in enumerate(os.listdir(folder_path), start=1):
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith(".html"):
        info = extract_info_from_html(file_path, index)
    elif file_name.endswith(".pdf"):
        info = extract_info_from_pdf(file_path, index)
    elif file_name.endswith(".csv"):
        info = extract_info_from_csv(file_path, index)
    else:
        continue

    chunks = split_into_chunks(info["content"])
    for chunk in chunks:
        data.append({"ID": f"{index}_{chunk_id}", "url": info["url"], "Title": info["Title"], "content": chunk})
        chunk_id += 1


df = pd.DataFrame(data)


dataset = Dataset.from_pandas(df)


dataset_dict = DatasetDict({"train": dataset})


login(token) 

try:
    dataset_dict.push_to_hub("SilvaFV/UFSCdatabase")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are logged in and have the correct permissions.")


from datasets import load_dataset
dataset = load_dataset("SilvaFV/UFSCdatabase")

print(dataset)

from sentence_transformers import SentenceTransformer
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", token=token)

def embed(batch):
    """
    Adds a column to the dataset called 'embeddings'
    """
    information = batch["content"]
    return {"embeddings": ST.encode(information)}

dataset = dataset.map(embed, batched=True, batch_size=500)
dataset.push_to_hub("SilvaFV/UFSCdatabase", revision="embedded")
print("Dataset with embeddings: ")
print(dataset)
