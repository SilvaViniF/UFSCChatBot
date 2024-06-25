import os
import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Define the folder containing the HTML files
folder_path = "files"
token = "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"
# Function to extract information from a single HTML file
def extract_info_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        
        # Extract ID from the file name or another source within the file
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Extract URL, Title, and content from the HTML
        url = soup.find("meta", property="og:url")["content"] if soup.find("meta", property="og:url") else "URL not found"
        title = soup.title.string if soup.title else "Title not found"
        content = soup.get_text(separator="\n").strip()
        
        return {"ID": file_id, "url": url, "Title": title, "content": content}

# Iterate over all HTML files in the folder and extract information
data = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".html"):
        file_path = os.path.join(folder_path, file_name)
        info = extract_info_from_html(file_path)
        data.append(info)

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Create a Hugging Face Dataset from the DataFrame
dataset = Dataset.from_pandas(df)

# Create a DatasetDict with the dataset
dataset_dict = DatasetDict({"train": dataset})

# Login to Hugging Face
login(token)  # Ensure you are logged in

# Push the dataset to the Hugging Face Hub
try:
    dataset_dict.push_to_hub("SilvaFV/UFSCdatabase")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you are logged in and have the correct permissions.")




#Embed Data
from datasets import load_dataset
dataset = load_dataset("SilvaFV/UFSCdatabase")

print(dataset)

from sentence_transformers import SentenceTransformer
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",token=token)


def embed(batch):
    """
    adds a column to the dataset called 'embeddings'
    """
    # or you can combine multiple columns here
    # For example the title and the text
    information = batch["content"]
    return {"embeddings" : ST.encode(information)}

dataset = dataset.map(embed,batched=True,batch_size=16)
dataset.push_to_hub("SilvaFV/UFSCdatabase", revision="embedded")
print("Dataset com embeddings: ")
print(dataset)
