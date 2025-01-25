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

def plot(graph):
    labels = {x: f"{graph.attribute(x, 'text')[:10]} ({x})" for x in graph.scan()}
    options = {
        "node_size": 750,
        "node_color": "#0277bd",
        "edge_color": "#454545",
        "font_color": "#fff",
        "font_size": 6,
        "alpha": 1.0
    }

    fig, ax = plt.subplots(figsize=(17, 8))
    pos = nx.spring_layout(graph.backend, seed=0, k=0.9, iterations=50)
    nx.draw_networkx(graph.backend, pos=pos, labels=labels, **options)
    ax.set_facecolor("#303030")
    ax.axis("off")
    fig.set_facecolor("#303030")

    plt.savefig("graph.png")

plot(graph)

for x in graph.showpath(0, 38):
    print(graph.node(x))
    
    
    
from txtai import LLM
import os
llm = LLM(os.getenv("MODEL_ID"))

def topic(graph):
  topic = list(graph.topics.keys())[0]

  text = "\n".join(graph.node(x)["text"] for x in graph.topics[topic])

  prompt = f"""<|im_start|>system
  You are a friendly assistant. You answer questions from users.<|im_end|>
  <|im_start|>user
  Label the following text with a topic name in a couple words.

  text: {text}
  topic: <|im_end|>
  <|im_start|>assistant
  """

  print(f"Topic: \"{topic}\", Generated Topic: \"{llm(prompt, maxlength=4096)}\"")

topic(graph)