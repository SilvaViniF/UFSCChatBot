import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import mmap

EMBEDDING_DIMENSION = 512  # Make sure this matches your embedding dimension

def load_embeddings(cache_file):
    """
    Load embeddings from a memory-mapped file.
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Embedding cache file not found: {cache_file}")

    with open(cache_file, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        embeddings = np.frombuffer(mm, dtype=np.float32).reshape(-1, EMBEDDING_DIMENSION)
    return embeddings

def visualize_embeddings(embeddings, labels=None, n_components=2, perplexity=30, n_iter=1000):
    """
    Visualize embeddings using t-SNE for dimensionality reduction.
    """
    # Reduce dimensionality
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    if n_components == 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], embeddings_2d[:, 2], 
                             c=labels if labels is not None else None, cmap='viridis')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_zlabel('t-SNE 3')
    else:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=labels if labels is not None else None, cmap='viridis')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    
    plt.title('t-SNE visualization of document embeddings')
    
    # Add color bar if labels are provided
    if labels is not None:
        plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig("embeddings_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Path to your embeddings cache file
    cache_file = 'cache_embeddings.mmap'
    
    # Load embeddings
    embeddings = load_embeddings(cache_file)
    
    # Create simple labels based on the index of each embedding
    # You might want to replace this with more meaningful labels if available
    labels = np.arange(len(embeddings))
    
    # Visualize embeddings
    visualize_embeddings(embeddings, labels=labels)
    
    # If you want a 3D plot:
    # visualize_embeddings(embeddings, labels=labels, n_components=3)