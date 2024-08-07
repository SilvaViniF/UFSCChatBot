from transformers import pipeline

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
chatbot = pipeline("text-generation",device_map="auto", model="mistralai/Mistral-Nemo-Instruct-2407",max_new_tokens=128)
chatbot(messages)
