import torch
import torch.nn as nnp
from transformers import AutoTokenizer
from huggingface_hub import login

def authenticate():
    login()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_capability(device))
print(torch.cuda.get_device_name(device))
##
authenticate()
##
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

documents = SimpleDirectoryReader("files").load_data()

from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 256
from llama_index.core import PromptTemplate


system_prompt = "Você é um assistente de perguntas e respostas. Seu objetivo é responder as perguntas de forma precisa, com base nas intruções e contexto oferecidos."

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
print(query_wrapper_prompt)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf" )


##
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm = HuggingFaceLLM(
    context_window=1024,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    model_name="meta-llama/Llama-2-7b-hf",
    device_map="auto",
    stopping_ids=stopping_ids,
    tokenizer_kwargs={"max_length": 1024},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

Settings.llm = llm
Settings.chunk_size = 256


##
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print(input)

def predict(input, history):
  response = query_engine.query(input)
  return str(response)

import gradio as gr

gr.ChatInterface(predict).launch(share=True)