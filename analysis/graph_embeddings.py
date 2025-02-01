from txtai import Embeddings


embeddings = Embeddings({
  "autoid": "uuid5",
  "path": "intfloat/e5-base",
  "instructions": {
    "query": "query: ",
    "data": "passage: "
  },
  "content": True,
  "graph": {
      "approximate": False,
      "topics": {}
  }
})

rag_embed = Embeddings()
rag_embed.load("app/cache/embeddings")

query =  """
SELECT id, text FROM txtai
order by percentile desc
LIMIT 100000
"""

embeddings.index(rag_embed.search(query))

graph = embeddings.search("UFSC Blumenau", 50, graph=True)

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

def plot(graph):
    labels = {x: f"{graph.attribute(x, 'id')[:10]} ({x})" for x in graph.scan()}
    
    backend = graph.backend
    degrees = dict(backend.degree())
    max_degree = max(degrees.values()) if degrees else 1
    node_colors = [cm.viridis(degrees[node] / max_degree) for node in backend.nodes()]
    
    pos = nx.spring_layout(backend, seed=42, k=0.9, iterations=100)

    fig, ax = plt.subplots(figsize=(20, 10))
    
    nx.draw_networkx_nodes(
        backend, pos,
        node_size=800,
        node_color=node_colors,
        alpha=0.9
    )
    
    nx.draw_networkx_edges(
        backend, pos,
        edge_color="#454545",
        width=1.5,
        alpha=0.8
    )
    
    nx.draw_networkx_labels(
        backend, pos,
        labels=labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        bbox=dict(facecolor="#303030", edgecolor="none", boxstyle="round,pad=0.2")
    )
    
    ax.set_axis_off()
    
    ax.set_facecolor("#303030")
    fig.set_facecolor("#ffffff")
    
    plt.title("Base de conhecimento UFSC Blumenau", fontsize=16, color="white", pad=20)
    
    plt.tight_layout()
    plt.savefig("graph.png", dpi=300)
    plt.show()

plot(graph)

for x in graph.showpath(0, 38):
    print(graph.node(x))
    
  