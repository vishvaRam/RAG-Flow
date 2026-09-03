from fastapi import APIRouter

from app.api.routes import agent, documents, health, history

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(agent.router)
api_router.include_router(documents.router)
api_router.include_router(history.router)
