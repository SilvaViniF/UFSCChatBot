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