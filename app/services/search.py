from txtai import Embeddings, LLM, RAG
from services.file_processing import get_documents
from config.settings import bnb_config
from dotenv import load_dotenv
import torch
import os
from config.settings import SYS_PROMPT, terminators
load_dotenv()

class SearchService:
    embeddings = None

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

    def talk(self, prompt: str, topn: int):
        rag = RAG(
            similarity=self.embeddings,
            path=llm,
            template=SYS_PROMPT,
            #minscore=0.8,
            #system=SYS_PROMPT,
            task="question-answering",
            context=topn
        )

        answer = rag(prompt,
        max_new_tokens=int(os.getenv('MAX_LENGTH')),
        truncation=True,
        eos_token_id=terminators,
        #do_sample=True,
        #temperature=1,
        #top_p=0.9,
        )
        return answer['answer']

llm = LLM(os.getenv("MODEL_ID"),
torch_dtype=torch.bfloat16,
device_map="auto",
quantization_config=bnb_config,
)

search_service = SearchService()