"""

from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

model_id = "v2ray/Llama-3-70B"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_input = data.get('input')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    response = pipeline(user_input)
    return jsonify({'response': response[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)
"""

from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("v2ray/Llama-3-70B")
input_text = [        
  'What is the capital of United States?'    
]
input_tokens = model.tokenizer(input_text,    
  return_tensors="pt",     
  return_attention_mask=False,     
  truncation=True,     
  max_length=MAX_LENGTH,     
  padding=False)

generation_output = model.generate(    
  input_tokens['input_ids'].cuda(),     
  max_new_tokens=20,    
  use_cache=True,    
  return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)