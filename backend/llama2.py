import torch
import torch.nn as nnp
from transformers import AutoTokenizer
from huggingface_hub import login

def authenticate():
    login()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

access_token = "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"
model = "meta-llama/Llama-2-7b-hf"
#model="meta-llama/Llama-2-7b-chat-hf"
#authenticate()
##
#import logging
#import sys

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings

documents = SimpleDirectoryReader("files/html_files").load_data()
"""
from llama_index.embeddings.fastembed import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512
"""






#system_prompt = "Você é um assistente de perguntas e respostas. Seu objetivo é responder as perguntas de forma precisa, com base nas intruções e contexto oferecidos."
system_prompt="You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
from llama_index.core import PromptTemplate
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")


tokenizer = AutoTokenizer.from_pretrained(model,token=access_token )

##
stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    #query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model,
    model_name=model,
    device_map="auto",
    #stopping_ids=stopping_ids,
    #tokenizer_kwargs={"max_length": 2048},
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}

)

#Settings.llm = llm
#Settings.chunk_size = 512


##
""""
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def predict(input, history):
  response = query_engine.query(input)
  return str(response)


import gradio as gr

gr.ChatInterface(predict).launch(share=True)
"""
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
embed_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
from llama_index.core import ServiceContext
service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)
index=VectorStoreIndex.from_documents(documents,service_context=service_context)
query_engine=index.as_query_engine()
while True:
    user_input = input("Enter your query: ")
    response = query_engine.query(user_input)
    print(response)
