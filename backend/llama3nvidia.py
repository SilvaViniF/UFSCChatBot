from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from datasets import load_dataset
import spaces
from threading import Thread
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time
#region Inicialização
app = Flask(__name__)
CORS(app)

token = ""
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("SilvaFV/UFSCdatabase",revision = "embedded")

data = dataset["train"]
data = data.add_faiss_index("embeddings") # Coluna do dataset com embeddings


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#endregion


#region Quantização
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
#endregion


#region Tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_id,token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    token=token
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
#endregion


#region Geração
SYS_PROMPT = """Você é um assistente para responder perguntas de alunos sobre a UFSC Blumenau.
Você recebe documentos e uma pergunta. Deve analisar a pergunta e responder com base nos documentos mais parecidos.
Suas respostas devem ser em portugues brasileiro.
Se você não souber a resposta, basta dizer “Essa informação não está disponível”. Não invente uma resposta."""

#Caso você saiba a resposta, você deve falar onde essa informação pode ser encontrada

def search(query: str, k: int = 5 ):
    """Uma função que faz embed tha nova query e retorna os resultados mais provaveis"""
    embedded_query = ST.encode(query)
    scores, retrieved_examples = data.get_nearest_examples(
        "embeddings", embedded_query, # comparando
        k=k # quantos resultados pegar
    )
    return scores, retrieved_examples

def format_prompt(prompt,retrieved_documents,k):
    """Usando os documentos fornecidos, vamos montar nosso contexto"""
    PROMPT = f"Pergunta:{prompt}\nContexto:"
    for idx in range(k) :
        PROMPT+= f"{retrieved_documents['content'][idx]}\n"
    return PROMPT

@spaces.GPU(duration=150) #max duration of talk
def talk(prompt):
    k = 5 # documentos recuparados
    scores , retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt,retrieved_documents,k)
    formatted_prompt = formatted_prompt[:3500] # evitar OOM
    print(formatted_prompt)
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
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
      temperature=0.6,
      top_p=0.9,
    )
    streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        temperature=0.6,
        eos_token_id=terminators,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
        print(outputs[-1])
    return outputs

#endregion


#region API
asda

@app.route("/api/userinput", methods=["POST"])
def user_input():
    prompt = request.json.get('message')
    ai_response = talk(prompt)
    response_list = list(ai_response)
    return jsonify({"response": response_list[-1]})

@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({'message': "Hello"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)

    #endregion 