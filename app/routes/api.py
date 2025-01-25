from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from services.search import SearchService
from models.search import SearchQuery

router = APIRouter()

search_service = SearchService()

@router.post("/query")
async def hybrid_search(query: SearchQuery):
    try:
        return StreamingResponse(
            search_service.talk(prompt=query.text, topn=query.top_n),
            media_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))