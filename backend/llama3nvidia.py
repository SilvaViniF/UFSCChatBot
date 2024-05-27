#region IMPORTS
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import json
import urllib.request
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Flask, jsonify,request,Response
#endregion

#region FLASK setup

#intanciar app:
app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True,port=8080)


#this method will get a string from the server
@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Hello"
    })
#this method will insert on the server
@app.route("api/userinput",methods=["POST"])
def user_input():
    d = request.json
    print(d)
    return Response(status=204)

#endregion

#region LLM SETUP

model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# Function to get vectorstore from URL
def get_vectorstore_from_url(url):
    with urllib.request.urlopen(url) as response:
        html_content = response.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ')
        return text
#endregion

#region Loading documents from URL
url = "https://blumenau.ufsc.br/"
chunk_list = get_vectorstore_from_url(url)
#endregion

#region format input/context
def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input
#endregion

#region Running retrieval
def retrieval(messages):
    formatted_query_for_retriever = '\n'.join([turn['role'] + ": " + turn['content'] for turn in messages]).strip()

    query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt')
    ctx_input = retriever_tokenizer(chunk_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

    ## Compute similarity scores using dot product and rank the similarity
    similarities = query_emb.matmul(ctx_emb.transpose(0, 1)) # (1, num_ctx)
    ranked_results = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)

    ## get top-n chunks (n=5)
    retrieved_chunks = [chunk_list[idx] for idx in ranked_results.tolist()[0][:5]]
    context = "\n\n".join(retrieved_chunks)

    ### running text generation
    formatted_input = get_formatted_input(messages, context)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
#endregion

#region conversation loop
while True:
    user_input = input("You: ")
    messages = [
        {"role": "user", "content": user_input}
    ]
    retrieval(messages)

#endregion
