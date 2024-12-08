from functools import partial
import multiprocessing
import re,fitz
from bs4 import BeautifulSoup
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig 
import torch 
import numpy as np
from flask import Flask, jsonify, request  
from flask_cors import CORS 
import spaces
import config,os,mmap,nltk,csv
from threading import Thread
from sentence_transformers import SentenceTransformer 
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import pickle
from typing import List
from functools import partial
import multiprocessing
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_DIMENSION = 512  # Dimension for mixedbread-ai/mxbai-embed-large-v1
# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp',quiet=True)

#app = Flask(__name__)
#CORS(app)


#region Processando Documentos
# Function to generate and cache embeddings using memory-mapped files
def generate_and_cache_embeddings(model: SentenceTransformer, documents: List[str], cache_file: str, mapping_file: str) -> np.ndarray:
    if not documents:
        print("No documents to process. Returning empty array.")
        return np.array([])

    embeddings = []
    with open(mapping_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['embedding_index', 'document_id', 'document_preview'])
        for idx, doc in enumerate(documents):
            embedding = model.encode(doc, convert_to_tensor=True)
            embeddings.append(embedding)
            writer.writerow([idx, f"doc_{idx}", doc[:50]])  # Write mapping info

    if not embeddings:
        print("No embeddings generated. Returning empty array.")
        return np.array([])

    embeddings_np = torch.stack(embeddings).cpu().numpy()

    with open(cache_file, 'wb') as f:
        f.write(embeddings_np.tobytes())

    return embeddings_np

def preprocess_chunk(chunk: str) -> str:
    # Remove excessive whitespace and non-printable characters
    return re.sub(r'\s+', ' ', ''.join(char for char in chunk if char.isprintable())).strip()



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

#ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=EMBEDDING_DIMENSION)
from pathlib import Path
folder_path = Path('/home/grupoh/backend/RAG_test')
file_path = [str(folder_path / f.name) for f in folder_path.iterdir() if f.is_file()]

documents = get_documents(file_path, cache_file='documents_cache.pkl')
cache_file = 'cache_embeddings.mmap'

document_embeddings = generate_and_cache_embeddings(ST, documents, cache_file,'embedding_mapping.csv')
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


#region busca híbrida
import faiss

#reranking:
from transformers import AutoTokenizer
import torch


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexHNSWFlat:
    vizinhos = 512
    # Hierarchical Navigable Small World (HSNW)
    index = faiss.IndexHNSWFlat(EMBEDDING_DIMENSION, vizinhos, faiss.METRIC_L2)
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings)
    return index

def preprocess_text(text: str) -> List[str]:
    tokens = word_tokenize(text.lower(), language='portuguese')
    stop_words = set(stopwords.words('portuguese') + ['é', 'são', 'está', 'estão', 'professor'])
    
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    #print(f"Filtered Tokens: {filtered_tokens}")
    
    return filtered_tokens
    
def initialize_bm25(documents: List[str]) -> BM25Okapi:
    tokenized_corpus = [preprocess_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

faiss_index = build_faiss_index(document_embeddings)

#remove filler words from docs
tokenized_corpus = [preprocess_text(doc) for doc in documents]

bm25 = bm25 = initialize_bm25(documents)

def search(query: str, model: SentenceTransformer, faiss_index: faiss.IndexHNSWFlat, bm25: BM25Okapi, k: int = 5) -> List[Tuple[int, float, str]]:
    query_embedding = model.encode(query,convert_to_tensor=True)
    query_embedding = query_embedding / torch.norm(query_embedding) #normalize query
    # Semantic search with FAISS
    D, I = faiss_index.search(query_embedding.unsqueeze(0).cpu().numpy(), k * 4)
    semantic_scores = 1 - np.sqrt(D[0]) / 2  # Convert L2 distance to similarity score
    semantic_indices = I[0]
    print(f"Semantic Scores: {semantic_scores}")
    print(f"Semantic Indices: {semantic_indices}")
    # Check the tokenization of the query
    tokenized_query = preprocess_text(query)
    print(f"Tokenized Query: {tokenized_query}")

    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    #print(f"Raw BM25 Scores - Max: {np.max(bm25_scores):.4f}, Mean: {np.mean(bm25_scores):.4f}, Non-zero: {np.count_nonzero(bm25_scores)}")
    
    # Combine results
    bm25_scores_rescored = bm25_scores[semantic_indices]
    # Normalize scores
    semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores) + 1e-8)
    bm25_scores_rescored = (bm25_scores_rescored - np.min(bm25_scores_rescored)) / (np.max(bm25_scores_rescored) - np.min(bm25_scores_rescored) + 1e-8)
    
    #combined_scores = 2 / (1/semantic_scores + 1/bm25_scores_rescored)
    combined_scores = 0.5 * semantic_scores + 0.5 * bm25_scores_rescored

    # Sort and get top-k
    top_k_indices = np.argsort(combined_scores)[::-1][:k]
    top_indices = semantic_indices[top_k_indices]
    top_scores = combined_scores[top_k_indices]

    results = []
    for idx, score in zip(top_indices, top_scores):
        doc = documents[int(idx)]
        # Check if the query terms are in the document
        if any(term in doc.lower() for term in query.lower().split()):
            results.append((int(idx), float(score), doc))
        if len(results) == k:
            break

    return results


#region Quantização
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
#endregion

#region Tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    token=config.token
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
#endregion

#region Geração
"""SYS_PROMPT = Você é um assistente para responder perguntas de alunos sobre a UFSC Blumenau.
Você recebe documentos relevantes e uma pergunta. Deve analisar a pergunta e responder com base nos documentos mais parecidos.
Suas respostas devem ser em português brasileiro, claras e concisas.
Mantenha a conversa em andamento, respondendo apenas à última pergunta recebida, mas levando em consideração o histórico da conversa para contexto adicional.
Se a pergunta não tiver relação com os documentos, ou se você não souber a resposta, basta dizer "Essa informação não está disponível". Não invente uma resposta.
Priorize informações precisas e úteis."""

SYS_PROMPT = """Você é um assistente para responder perguntas de alunos sobre a UFSC Blumenau.
Você recebe um contexto relevante e uma pergunta. Deve analisar a pergunta e responder com base no contexto, ignorando informações que não tenham relação com a pergunta.
Suas respostas devem ser em português brasileiro, claras e concisas.
Se a pergunta não tiver relação com os documentos, ou se você não souber a resposta, basta dizer "Essa informação não está disponível". Não invente uma resposta.
Priorize informações precisas e úteis.
Não repita a pergunta na sua resposta, apenas a responda."""

chat_history = []

def format_prompt(prompt, retrieved_documents, k):
    print("Montando contexto")
    PROMPT = f"Contexto:\n"
    # If retrieved_documents is a list of dicts, modify the loop
    for idx, doc in enumerate(retrieved_documents[:k]):
        PROMPT += f"ID: {doc.get('ID', idx)}\n"
        PROMPT += f"Título: {doc.get('Title', 'Sem título')}\n"
        PROMPT += f"Conteúdo: {doc.get('content', 'Sem conteúdo')}\n"
    return PROMPT

@spaces.GPU(duration=150)  # max duration of talk
def talk(prompt, max_new_tokens=1024):
    print("Pensando")
    k = 20  # Increase the number of retrieved documents
    try:
        retrieved_documents = search(prompt, ST, faiss_index, bm25, k=k)
    except Exception as e:
        print(f"Error in search function: {e}")
        return []

    """for idx, (doc_idx, score, doc) in enumerate(retrieved_documents, 1):
        print(f"Result {idx}:")
        print(f"  Document chunk {doc_idx + 1}")
        print(f"  Relevance Score: {score:.4f}")
        print(f"  Content: {doc[:200]}...")
        print("") """

    top_context = "\n".join([doc for _, _, doc in retrieved_documents[:5]])
    #print(f"TOP CONTEXT: {top_context}")


    global chat_history
    #history_text = "\n".join([f"Pergunta: {h['user']}\nResposta: {h['assistant']}" for h in chat_history])
    #Disabled conversation chain history for testing in judge.py
    history_text=""
    complete_prompt = f"{history_text}\nPergunta: {prompt}\n{top_context}"
    #debug:
    #print(complete_prompt)

    messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": complete_prompt}]
    
    # geração
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1,
        top_p=0.9,
    )
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.3,
        eos_token_id=terminators,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
    
    #form_output = ''.join(filter(None, outputs))
    #chat_history.append({"user": prompt, "assistant": form_output})
    #if len(chat_history) > 3:
    #    chat_history.pop(0)
    
    return outputs


@spaces.GPU(duration=150)
def eval(prompt):
    print("Evaluating")
    
    #debug:
    #print(prompt)

    messages = [{"role": "system", "content":""}, {"role": "user", "content": prompt}]
    
    # geração
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
    )
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.3,
        eos_token_id=terminators,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
    
    return outputs

#endregion

#region API

#@app.route("/api/userinput", methods=["POST"])
def user_input():
    prompt = request.json.get('message')
    ai_response = talk(prompt)
    response_list = list(ai_response)
    return jsonify({"response": response_list[-1] if response_list else ""})

#@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({'message': "Hello"})

#if __name__ == "__main__":
#  app.run(debug=True, host='0.0.0.0', port=3000)

#endregion