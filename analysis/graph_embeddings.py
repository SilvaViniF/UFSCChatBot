import os
from txtai import Embeddings
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
import platform

if platform.processor() == "arm":
    os.environ["OMP_NUM_THREADS"] = "1"

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

query = """
SELECT id, text FROM txtai
order by percentile desc
LIMIT 100000
"""
embeddings.index(rag_embed.search(query))

graph = embeddings.search("UFSC Blumenau", 50, graph=True)


def plot(graph):
    labels = {}
    for x in graph.scan():
        topic = graph.attribute(x, "refined_topic")
        
        if not topic or len(topic) > 500:
            topic = f"{graph.attribute(x, 'topic')[:10]}"
        if ':' in topic:
            topic = topic.split(':', 1)[1].strip()
            
        labels[x] = f"{topic} ({x})"

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
        width=0.5,
        alpha=0.2
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


def refine_topics(graph):
    refined_topics = {}
    for original_topic in graph.topics.keys():
        combined_text = "\n".join(graph.attribute(uid, "text") for uid in graph.topics[original_topic])
        
        prompt = f"""<|im_start|>system
You are a friendly assistant. Your task is to generate a concise, descriptive topic label in just a few words.
DO NOT USE MORE THAN 5 WORDS!!
<|im_end|>
<|im_start|>user
Based on the following texts, please generate a refined topic label IN BRAZILIAN PORTUGUESE!:
{combined_text}
Refined topic label: <|im_end|>
<|im_start|>assistant"""
        
        from app.services.search import SearchService
        service = SearchService()
        messages = [{"role": "system", "content": prompt}]
        
        generator_output = service.generate(messages)
        refined_label = ''.join(part for part in generator_output).strip()
        
        refined_topics[original_topic] = refined_label
        
        print(f"Original Topic Key: {original_topic}")
        print(f"Refined Topic Label: {refined_label}\n")
        
    return refined_topics



refined = refine_topics(graph)


for original_topic, node_ids in graph.topics.items():
    refined_label = refined.get(original_topic)
    if refined_label:
        for uid in node_ids:
            node = graph.node(uid)
            node["refined_topic"] = refined_label

plot(graph)
