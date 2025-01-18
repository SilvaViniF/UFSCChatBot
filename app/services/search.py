from txtai import Embeddings
from services.file_processing import get_documents
from config.settings import bnb_config
from dotenv import load_dotenv
import torch
import os
from config.settings import SYS_PROMPT, terminators
from threading import Thread
from rank_bm25 import BM25Okapi
import nltk
import pickle
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('rslp',quiet=True)
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from nltk.tokenize import word_tokenize,
import unicodedata
from nltk.corpus import stopwords


load_dotenv()
files_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../app/files"))

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
    chunk_list = None  # Initialize chunk_list as None
    stop_words = set(stopwords.words('portuguese') + 
        ['é', 'são', 'está', 'estão', 'professor', 
        'ser', 'estar', 'ter', 'haver', 'fazer'])

    @classmethod
    def set_embeddings(cls, embeddings_param: Embeddings):
        cls.embeddings = embeddings_param

    def index_chunks(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not set. Please set embeddings before indexing.")

        if self.embeddings.exists("test"):
            self.embeddings.load("test")  # TODO change for .env
        else:
            chunk_list = get_documents(files_dir)
            self.embeddings.index(chunk_list)
            self.embeddings.save("test")
            # Save chunk_list to a pickle file
            with open('chunk_list.pickle', 'wb') as f:
                pickle.dump(chunk_list, f)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with Portuguese-specific handling.
        Returns preprocessed text as string for embedding, but uses tokens for matching.
        """
        # Normalize unicode characters (handle accents)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize with Portuguese-specific handling
        tokens = word_tokenize(text, language='portuguese')
        
        # Filter tokens
        filtered_tokens = [
            token for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        # Return processed text as string for embedding
        return ' '.join(filtered_tokens)
    
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
        processed_prompt = self.preprocess_text(prompt)
        prompt_tokens = self.get_tokens(prompt)
        
        # Get initial candidates
        retrieved = self.embeddings.search(processed_prompt, limit=topn * 4)
        
        results = []
        seen_content = set()  # Track duplicate content
        
        for item in retrieved:
            idx = item["id"]
            score = item["score"]
            doc = item["text"]
            
            # Process document text
            doc_processed = self.preprocess_text(doc)
            if doc_processed in seen_content:
                continue
                
            # Get document tokens for keyword matching
            doc_tokens = self.get_tokens(doc)
            
            # Hybrid scoring with Portuguese-specific token matching
            keyword_score = self._calculate_keyword_score(prompt_tokens, doc_tokens)
            combined_score = 0.7 * score + 0.3 * keyword_score
            
            results.append((int(idx), float(combined_score), doc))
            seen_content.add(doc_processed)
            
            if len(results) == topn:
                break
        
        # Sort by combined score
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
            # Calculate content relevance features
            length_score = self._normalize_length_score(doc)
            density_score = self._calculate_information_density(doc)
            
            # Calculate token overlap density
            doc_tokens = self.get_tokens(doc)
            token_overlap = len(set(prompt_tokens) & set(doc_tokens)) / len(prompt_tokens) if prompt_tokens else 0
            
            # Combine scores with weights
            final_score = (
                0.5 * score +  # Original similarity
                0.2 * length_score +  # Length appropriateness
                0.2 * density_score +  # Information density
                0.1 * token_overlap  # Token overlap density
            )
            
            reranked.append((idx, final_score, doc))
        
        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def _normalize_length_score(self, doc: str) -> float:
        """Calculate normalized score based on chunk length."""
        optimal_length = 768  # Adjust based on your needs
        current_length = len(doc.split())
        return 1.0 / (1.0 + abs(current_length - optimal_length) / optimal_length)

    def _calculate_information_density(self, doc: str) -> float:
        """Calculate information density score with Portuguese-specific handling."""
        # Tokenize with Portuguese-specific handling
        doc_tokens = self.get_tokens(doc)
        
        if not doc_tokens:
            return 0.0
            
        # Calculate unique tokens ratio
        unique_tokens = len(set(doc_tokens))
        total_tokens = len(doc_tokens)
        
        return unique_tokens / total_tokens if total_tokens > 0 else 0.0

    def talk(self, prompt: str, topn: int):
        initial_results = self.search(prompt,topn)
        reranked_results = self.rerank_results(prompt, initial_results)
        
        context_pieces = []
        for _, _, doc in reranked_results:
            context_pieces.append(doc.strip())
        
        top_context = "\n\n---\n\n".join(context_pieces)
        
        #history_text = "\n".join([f"Pergunta: {h['user']}\nResposta: {h['assistant']}" for h in self.chat_history])
        history_text =""
        complete_prompt = f"{history_text}\nPergunta: {prompt}\n Contexto:{top_context}"
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
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
        )
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
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