from fastapi import APIRouter, HTTPException
from services.search import SearchService
from models.search import SearchQuery, SearchResults

router = APIRouter()

search_service = SearchService()

@router.post("/query")
async def hybrid_search(query: SearchQuery):
    try:
        results = search_service.talk(prompt=query.text, topn=query.top_n)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# #@app.route("/api/userinput", methods=["POST"])
# def user_input():
#     prompt = request.json.get('message')
#     ai_response = talk(prompt)
#     response_list = list(ai_response)
#     return jsonify({"response": response_list[-1] if response_list else ""})
