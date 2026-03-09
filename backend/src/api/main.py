from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import query, ingest
from src.api.dependencies import get_pipeline

app = FastAPI(
    title="Modular RAG API",
    version="1.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(ingest.router)

@app.get("/health")
def health(pipeline=Depends(get_pipeline)):
    return {"status": "up", "stats": pipeline.get_stats()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
