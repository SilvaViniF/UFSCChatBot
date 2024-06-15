from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import json
import urllib.request
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Lock

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize a lock for thread safety
lock = Lock()

# Load the models once when the application starts
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')
eos_token_id = tokenizer.eos_token_id
pad_token_id = eos_token_id

# Function to get text from URL
def get_text_from_url(url):
    with urllib.request.urlopen(url) as response:
        html_content = response.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ')

# Load document text from URL
url = "https://blumenau.ufsc.br/"
chunk_list = get_text_from_url(url)

# Format input function
def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(
        ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input

# Retrieval function
def retrieval(messages):
    torch.cuda.empty_cache()  # Clear CUDA cache
    formatted_query_for_retriever = '\n'.join([turn['role'] + ": " + turn['content'] for turn in messages]).strip()

    query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt', max_length=128, truncation=True)
    ctx_input = retriever_tokenizer(chunk_list, padding=True, truncation=True, max_length=128, return_tensors='pt')
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

    similarities = query_emb.matmul(ctx_emb.transpose(0, 1))
    ranked_results = torch.argsort(similarities, dim=-1, descending=True)
    retrieved_chunks = [chunk_list[idx] for idx in ranked_results.tolist()[0][:5]]
    context = "\n\n".join(retrieved_chunks)

    formatted_input = get_formatted_input(messages, context)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt", max_length=256, truncation=True).to(model.device)

    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model.generate(
            input_ids=tokenized_prompt.input_ids,
            attention_mask=tokenized_prompt.attention_mask,
            max_new_tokens=64,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# API endpoint for user input
@app.route("/api/userinput", methods=["POST"])
def user_input():
    user_message = request.json.get('message')
    messages = [{"role": "user", "content": user_message}]
    
    # Call retrieval outside the API call context
    with lock:
        ai_response = retrieval(messages)
    
    return jsonify({"response": ai_response})

# Home endpoint
@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({'message': "Hello"})

# Main function
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
