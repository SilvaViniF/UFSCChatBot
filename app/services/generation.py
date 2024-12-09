from search import search
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    token=config.token
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

chat_history = []

def format_prompt(prompt, retrieved_documents, k):
    print("Montando contexto")
    PROMPT = f"Contexto:\n"
    # If retrieved_documents is a list of dicts, modify the loop
    for idx, doc in enumerate(retrieved_documents[:k]):
        PROMPT += f"ID: {doc.get('ID', idx)}\n"
        PROMPT += f"Título: {doc.get('Title', 'Sem título')}\n"
        PROMPT += f"Conteúdo: {doc.get('content', 'Sem conteúdo')}\n"
    return PROMPT


def talk(prompt: str, top_k: int=20):

    try:
        retrieved_documents = search(prompt, ST, faiss_index, bm25, k=top_k)
    except Exception as e:
        print(f"Error in search function: {e}")
        return []

    """for idx, (doc_idx, score, doc) in enumerate(retrieved_documents, 1):
        print(f"Result {idx}:")
        print(f"  Document chunk {doc_idx + 1}")
        print(f"  Relevance Score: {score:.4f}")
        print(f"  Content: {doc[:200]}...")
        print("") """

    top_context = "\n".join([doc for _, _, doc in retrieved_documents[:5]])
    #print(f"TOP CONTEXT: {top_context}")


    global chat_history
    #history_text = "\n".join([f"Pergunta: {h['user']}\nResposta: {h['assistant']}" for h in chat_history])
    #Disabled conversation chain history for testing in judge.py
    history_text=""
    complete_prompt = f"{history_text}\nPergunta: {prompt}\n{top_context}"
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
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1,
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
    
    #form_output = ''.join(filter(None, outputs))
    #chat_history.append({"user": prompt, "assistant": form_output})
    #if len(chat_history) > 3:
    #    chat_history.pop(0)
    
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
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.5,
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

