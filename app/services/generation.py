from search import search
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_ID = os.getenv('MODEL_ID')
