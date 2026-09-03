from fastapi import APIRouter

from app.core.database import db_manager

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    db_ok = False
    try:
        async with db_manager.acquire_pg() as conn:
            val = await conn.fetchval("SELECT 1")
            db_ok = val == 1
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
    }
