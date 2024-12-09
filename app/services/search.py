from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

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