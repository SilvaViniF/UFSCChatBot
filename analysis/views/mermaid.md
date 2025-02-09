

```mermaid
sequenceDiagram
    participant User as Frontend
    participant FileProcessing as Processamento de arquivos
    participant Embeddings as Embeddings
    participant SearchService as Serviço de busca
    participant BM25 as BM25
    participant EmbeddingsSearch as Busca de embeddings
    participant Reranking as Reranking
    participant LLM as LLM
    participant ChatHistory as Histórico

    User ->> SearchService: Envia uma pergunta (prompt)
    activate SearchService

    SearchService ->> FileProcessing: Solicita documentos processados
    activate FileProcessing
    FileProcessing -->> SearchService: Retorna lista de Chunks
    deactivate FileProcessing

    SearchService ->> Embeddings: Realiza busca semântica
    activate Embeddings
    Embeddings -->> SearchService: Retorna resultados da busca
    deactivate Embeddings

    SearchService ->> BM25: Realiza busca por palavras-chave
    activate BM25
    BM25 -->> SearchService: Retorna resultados da busca
    deactivate BM25

    SearchService ->> Reranking: Reorganiza resultados
    activate Reranking
    Reranking -->> SearchService: Retorna resultados rerankeados
    deactivate Reranking

    SearchService ->> LLM: Constrói e envia o prompt
    activate LLM
    LLM -->> SearchService: Gera e retorna a resposta
    deactivate LLM

    SearchService ->> ChatHistory: Armazena a interação
    activate ChatHistory
    ChatHistory -->> SearchService: Confirma armazenamento
    deactivate ChatHistory

    SearchService -->> User: Retorna a resposta final
    deactivate SearchService