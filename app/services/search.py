from txtai import Embeddings, LLM, RAG
from services.file_processing import get_documents
from config.settings import bnb_config
from dotenv import load_dotenv
import torch
import os

load_dotenv()

def index_chunks(embeddings: Embeddings):
    
    if embeddings.exists("test"):
        embeddings.load("test") #TODO change for .env
    else:
        chunk_list = get_documents(os.getenv('FILES'),int(os.getenv('MAX_LENGTH')))
        embeddings.index(chunk_list)
        embeddings.save("test")
    
def talk(prompt: str):
    
    embeddings = Embeddings(content=True)
    index_chunks(embeddings)

    llm = LLM(os.getenv("MODEL_ID"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config
    )

    rag = RAG(embeddings, llm, template=prompt)

    answer = rag(prompt)
    
    return answer['answer']