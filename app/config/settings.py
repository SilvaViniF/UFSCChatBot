from pydantic import BaseSettings

class Settings(BaseSettings):
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    faiss_dimension: int = 512
    stopwords_language: str = "portuguese"

settings = Settings()


#region Geração
"""SYS_PROMPT = Você é um assistente para responder perguntas de alunos sobre a UFSC Blumenau.
Você recebe documentos relevantes e uma pergunta. Deve analisar a pergunta e responder com base nos documentos mais parecidos.
Suas respostas devem ser em português brasileiro, claras e concisas.
Mantenha a conversa em andamento, respondendo apenas à última pergunta recebida, mas levando em consideração o histórico da conversa para contexto adicional.
Se a pergunta não tiver relação com os documentos, ou se você não souber a resposta, basta dizer "Essa informação não está disponível". Não invente uma resposta.
Priorize informações precisas e úteis."""

SYS_PROMPT = """Você é um assistente para responder perguntas de alunos sobre a UFSC Blumenau.
Você recebe um contexto relevante e uma pergunta. Deve analisar a pergunta e responder com base no contexto, ignorando informações que não tenham relação com a pergunta.
Suas respostas devem ser em português brasileiro, claras e concisas.
Se a pergunta não tiver relação com os documentos, ou se você não souber a resposta, basta dizer "Essa informação não está disponível". Não invente uma resposta.
Priorize informações precisas e úteis.
Não repita a pergunta na sua resposta, apenas a responda."""

#ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=EMBEDDING_DIMENSION)
from pathlib import Path
folder_path = Path('/home/grupoh/backend/RAG_test')
file_path = [str(folder_path / f.name) for f in folder_path.iterdir() if f.is_file()]

documents = get_documents(file_path, cache_file='documents_cache.pkl')
cache_file = 'cache_embeddings.mmap'

document_embeddings = generate_and_cache_embeddings(ST, documents, cache_file,'embedding_mapping.csv')
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

