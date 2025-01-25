from txtai import Embeddings
from .file_processing import get_documents
from dotenv import load_dotenv
import torch
import os
from config.settings import SYS_PROMPT, terminators, bnb_config
from threading import Thread
from rank_bm25 import BM25Okapi
import nltk
import pickle
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp',quiet=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from nltk.tokenize import word_tokenize
import unicodedata
from nltk.corpus import stopwords


load_dotenv()
files_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../app/files"))
cache_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../app/cache"))
embeddings_db = os.path.join(cache_dir, "embeddings")
chunk_cache = os.path.join(cache_dir, "chunk_cache")

class SearchService:
    embeddings = None
    chat_history = []
    token = ""
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
    chunk_list = None
    stop_words = set(stopwords.words('portuguese') + 
        ['é', 'são', 'está', 'estão', 'professor', 
        'ser', 'estar', 'ter', 'haver', 'fazer'])

    @classmethod
    def set_embeddings(cls, embeddings_param: Embeddings):
        cls.embeddings = embeddings_param

    def index_chunks(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not set. Please set embeddings before indexing.")

        if self.embeddings.exists(embeddings_db):
            self.embeddings.load(embeddings_db)
        else:
            chunk_list = get_documents(files_dir, int(os.getenv("MAX_LENGTH")))
            self.embeddings.index(chunk_list)
            self.embeddings.save(embeddings_db)
            with open(chunk_cache, 'wb') as f:
                pickle.dump(chunk_list, f)
    
    def get_tokens(self, text: str) -> list[str]:
        """Get filtered tokens for keyword matching."""
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        tokens = word_tokenize(text.lower(), language='portuguese')
        return [token for token in tokens if token.isalnum() and token not in self.stop_words]
    
    def initialize_bm25(self, documents: list[list[str]]) -> BM25Okapi:
        squashed_documents = [" ".join(doc) for doc in documents]
        tokenized_corpus = [self.preprocess_text(doc) for doc in squashed_documents]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def search(self, prompt: str, topn: int):
        prompt_tokens = self.get_tokens(prompt)
        processed_prompt = " ".join(prompt_tokens)
        retrieved = self.embeddings.search(processed_prompt, limit=topn * 4)
        
        results = []
        seen_content = set()
        
        for item in retrieved:
            idx = item["id"]
            score = item["score"]
            doc = item["text"]
            doc_tokens = self.get_tokens(doc)
            doc_processed = " ".join(doc_tokens)
            
            if doc_processed in seen_content:
                continue
            
            keyword_score = self._calculate_keyword_score(prompt_tokens, doc_tokens)
            combined_score = 0.7 * score + 0.3 * keyword_score
            results.append((int(idx), float(combined_score), doc))
            seen_content.add(doc_processed)
            
            if len(results) == topn:
                break
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_keyword_score(self, prompt_tokens: list[str], doc_tokens: list[str]) -> float:
        """Calculate keyword matching score using preprocessed tokens."""
        if not prompt_tokens:
            return 0.0
        prompt_token_set = set(prompt_tokens)
        doc_token_set = set(doc_tokens)
        matches = prompt_token_set.intersection(doc_token_set)
        return len(matches) / len(prompt_token_set)
    
    def rerank_results(self, prompt: str, initial_results: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
        """Rerank results based on additional criteria."""
        reranked = []
        prompt_tokens = self.get_tokens(prompt)
        
        for idx, score, doc in initial_results:
            length_score = self._normalize_length_score(doc)
            density_score = self._calculate_information_density(doc)
            doc_tokens = self.get_tokens(doc)
            token_overlap = len(set(prompt_tokens) & set(doc_tokens)) / len(prompt_tokens) if prompt_tokens else 0
            
            final_score = (
                0.5 * score +
                0.2 * length_score +
                0.2 * density_score +
                0.1 * token_overlap
            )
            
            reranked.append((idx, final_score, doc))
        
        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def _normalize_length_score(self, doc: str) -> float:
        """Calculate normalized score based on chunk length."""
        optimal_length = int(os.getenv("MAX_LENGTH"))
        current_length = len(doc.split())
        return 1.0 / (1.0 + abs(current_length - optimal_length) / optimal_length)

    def _calculate_information_density(self, doc: str) -> float:
        """Calculate information density score with Portuguese-specific handling."""
        doc_tokens = self.get_tokens(doc)
        if not doc_tokens:
            return 0.0
        unique_tokens = len(set(doc_tokens))
        total_tokens = len(doc_tokens)
        return unique_tokens / total_tokens if total_tokens > 0 else 0.0

    def talk(self, prompt: str, topn: int, sys_prompt: str = SYS_PROMPT):
        initial_results = self.search(prompt, topn)
        reranked_results = self.rerank_results(prompt, initial_results)

        context_pieces = [doc.strip() for _, _, doc in reranked_results]
        top_context = "\n\n---\n\n".join(context_pieces)
        history_text = ""
        complete_prompt = f"{history_text}\nPergunta: {prompt}\n Contexto:{top_context}"

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": complete_prompt}]

  
        for chunk in self.generate(messages):
            yield chunk

        form_output = "".join(chunk for chunk in self.generate(messages))
        self.chat_history.append({"user": prompt, "assistant": form_output})
        if len(self.chat_history) > 3:
            self.chat_history.pop(0)

        
    def generate(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )

        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=4096,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for text in streamer:
            yield text


search_service = SearchService()