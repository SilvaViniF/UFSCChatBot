from txtai import Embeddings, LLM, RAG
from services.file_processing import get_documents
from config.settings import bnb_config
from dotenv import load_dotenv
import torch
import os
from config.settings import SYS_PROMPT, terminators
from sentence_transformers import SentenceTransformer
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
load_dotenv()

class SearchService:
    embeddings = None
    chat_history = []
    token = ""
    ST = SentenceTransformer("sentence-transformers/nli-mpnet-base-v2")
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_ID"), token=token)
    model = AutoModelForCausalLM.from_pretrained(
        os.getenv("MODEL_ID"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        token=token
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    @classmethod
    def set_embeddings(cls, embeddings_param: Embeddings):
        cls.embeddings = embeddings_param

    def index_chunks(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not set. Please set embeddings before indexing.")

        if self.embeddings.exists("test"):
            self.embeddings.load("test")  # TODO change for .env
        else:
            chunk_list = get_documents(os.getenv('FILES'))
            self.embeddings.index(chunk_list)
            self.embeddings.save("test")

    def search(self, prompt: str, topn: int):
        results = self.embeddings.search(prompt,limit=topn)
        return results

    def talk(self, prompt: str, topn: int):
        retrieved_chunks = self.search(prompt,topn)
        top_context ="\n".join([doc["text"] for doc in retrieved_chunks])
        
        history_text = "\n".join([f"Pergunta: {h['user']}\nResposta: {h['assistant']}" for h in self.chat_history])
        complete_prompt = f"{history_text}\nPergunta: {prompt}\n{top_context}"
        #debug:
        print(complete_prompt)

        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": complete_prompt}]
        
        # geração
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=1,
            top_p=0.9,
        )
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
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
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield "".join(outputs)
        
        form_output = ''.join(filter(None, outputs))
        self.chat_history.append({"user": prompt, "assistant": form_output})
        if len(self.chat_history) > 3:
            self.chat_history.pop(0)
        
        return outputs

search_service = SearchService()