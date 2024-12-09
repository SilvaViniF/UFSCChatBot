from fastapi import APIRouter, HTTPException
from services.generation import talk
from app.models.search import SearchQuery,SearchResults

router = APIRouter()

@router.post("/query", response_model=SearchResults)
async def hybrid_search(query: SearchQuery):
    try:
        results = talk(query.text, query.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# #@app.route("/api/userinput", methods=["POST"])
# def user_input():
#     prompt = request.json.get('message')
#     ai_response = talk(prompt)
#     response_list = list(ai_response)
#     return jsonify({"response": response_list[-1] if response_list else ""})
