from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig 
import torch 
from flask import Flask, jsonify, request  
from flask_cors import CORS 
from datasets import load_dataset 
import spaces 
from threading import Thread
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
#region Inicialização
app = Flask(__name__)
CORS(app)

token = ""
ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

dataset = load_dataset("SilvaFV/UFSCdatabase", revision="embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings")  # Coluna do dataset com embeddings

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
#endregion

#region Quantização
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
#endregion

#region Tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Change this to a custom device map if needed
    quantization_config=bnb_config,
    # load_in_8bit_fp32_cpu_offload=True,  # Remove this line
    token=token
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
#endregion


SYS_PROMPT = """Você é um assistente para responder perguntas de alunos sobre a UFSC Blumenau.
Você recebe um contexto relevante e uma pergunta. Deve analisar a pergunta e responder com base no contexto, ignorando informações que não tenham relação com a pergunta.
Suas respostas devem ser em português brasileiro, claras e concisas.
Se a pergunta não tiver relação com os documentos, ou se você não souber a resposta, basta dizer "Essa informação não está disponível". Não invente uma resposta.
Priorize informações precisas e úteis.
Não repita a pergunta na sua resposta, apenas a responda."""

chat_history = []


import numpy as np

from sentence_transformers import CrossEncoder

# Initialize the cross-encoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def search(query: str, k: int = 20):
    embedded_query = ST.encode(query)
    
    # Cosine Similarity Search
    similarities = cosine_similarity(embedded_query.reshape(1, -1), data["embeddings"])
    indexed = list(enumerate(similarities[0]))
    sorted_cosine_index = sorted(indexed, key=lambda x: x[1], reverse=True)
    
    # Retrieve top documents using Cosine Similarity
    cosine_documents = [
        {"content": data[i]['content'], "Title": data[i]['Title'], "score": score}
        for i, score in sorted_cosine_index if score > 0.4
    ]
    
    # FAISS Search with Cross-Encoder Reranking
    initial_k = k * 3  # Retrieve more documents initially
    faiss_scores, faiss_examples = data.get_nearest_examples(
        "embeddings", embedded_query, k=initial_k
    )
    
    # Rerank using Cross-Encoder
    pairs = [(query, doc) for doc in faiss_examples['content']]
    rerank_scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(rerank_scores)[-k:][::-1]

    faiss_documents = [
        {"content": faiss_examples['content'][i], "Title": faiss_examples['Title'][i], "score": rerank_scores[i]}
        for i in top_indices
    ]
    
    # Normalize scores
    cosine_max = max([doc['score'] for doc in cosine_documents]) if cosine_documents else 1
    faiss_max = max([doc['score'] for doc in faiss_documents]) if faiss_documents else 1
    
    for doc in cosine_documents:
        doc['normalized_score'] = doc['score'] / cosine_max
    for doc in faiss_documents:
        doc['normalized_score'] = doc['score'] / faiss_max
    
    # Merge and sort results by normalized score
    combined_documents = cosine_documents + faiss_documents
    combined_documents = sorted(combined_documents, key=lambda x: x['normalized_score'], reverse=True)

    # Return top k documents
    return combined_documents[:k]


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
def talk(prompt,max_new_tokens=512):
    print("Pensando")
    k =10  # documentos recuparados
    
    retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt, retrieved_documents, k)
    formatted_prompt = formatted_prompt[:1000]  # evitar OOM

    
    global chat_history
    #history_text = "\n".join([f"Pergunta: {h['user']}\nResposta: {h['assistant']}" for h in chat_history])
    #Disabled conversation chain history for testing in judge.py
    history_text=""
    complete_prompt = f"{history_text}\nPergunta: {prompt}\n{formatted_prompt}"
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
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,
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
    
    form_output = ''.join(filter(None, outputs))
    chat_history.append({"user": prompt, "assistant": form_output})
    if len(chat_history) > 3:
        chat_history.pop(0)
    
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
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.3,
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
@app.route("/api/userinput", methods=["POST"])
def user_input():
    prompt = request.json.get('message')
    ai_response = talk(prompt)
    response_list = list(ai_response)
    return jsonify({"response": response_list[-1] if response_list else ""})

@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({'message': "Hello"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)

#endregion