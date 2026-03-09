from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas.request import QueryRequest
from src.api.dependencies import get_pipeline

router = APIRouter(prefix="/query", tags=["RAG"])
print("📡 Query router ready")  # Log pour vérifier que le routeur est chargé


@router.post("")
async def ask_question(req: QueryRequest, pipeline = Depends(get_pipeline)):
    try:
        print(f"Received query request: {req.dict()}")
        # Appel du pipeline en transmettant l'historique de chat au LLM via `llm_params`
        llm_params = req.llm_params or {}
        # Le frontend envoie `chat_history` sous forme de liste d'objets {role, content}
        if req.chat_history:
            llm_params = {**llm_params, 'conversation_history': req.chat_history}

        response = pipeline.query(
            query_text=req.question,
            top_k=req.top_k,
            rerank_top_k=req.rerank_top_k,
            llm_params=llm_params
        )

        # `response` est un `RAGResponse` — convertir en JSON simple
        return response.to_dict()

    except Exception as e:
        print(f"Error during query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur Query: {str(e)}")