from fastapi import FastAPI
from routes import api
import uvicorn
from txtai import Embeddings
import services.search
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

def create_app() -> FastAPI:

    app = FastAPI(title="RAG API", version="1.0.0")
    
    app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

    app.include_router(api.router, prefix="/search", tags=["Search"])
    embeddings = Embeddings(content=True, path="mixedbread-ai/mxbai-embed-large-v1")
    services.search.SearchService.set_embeddings(embeddings)

    search_service = services.search.SearchService()
    search_service.index_chunks()

    @app.get("/")
    async def root():
        return {"message": "Bem vindo ao Chat Bot UFSC - Blumenau!"}
    
    return app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, reload=False)