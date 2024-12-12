from fastapi import FastAPI
from routes import api
import uvicorn

def create_app() -> FastAPI:

    app = FastAPI(title="RAG API", version="1.0.0")

    app.include_router(api.router, prefix="/search", tags=["Search"])

    @app.get("/")
    async def root():
        return {"message": "Bem vindo ao Chat Bot UFSC - Blumenau!"}
    
    return app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, reload=False)
