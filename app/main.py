from fastapi import FastAPI
from app.routes import api

app = FastAPI(title="RAG API", version="1.0.0")

app.include_router(api.router, prefix="/search", tags=["Search"])

@app.get("/")
async def root():
    return {"message": "Bem vindo ao Chat Bot UFSC - Blumenau!"}
