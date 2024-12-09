from pydantic import BaseModel

class SearchQuery(BaseModel):
    text: str
    top_k: int = 5


class SearchResult(BaseModel):
    document_id: int
    score: float
    content: str

class SearchResults(BaseModel):
    results: list[SearchResult]
