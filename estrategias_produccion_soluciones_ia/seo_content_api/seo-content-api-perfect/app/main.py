from fastapi import FastAPI

from app.routers import articles, faqs, keywords, metadata, social

app = FastAPI(
    title="SEO Content API",
    description="API REST con FastAPI y Azure OpenAI para generación de contenido SEO con IA.",
    version="1.0.0",
)

app.include_router(keywords.router, prefix="/api/keywords")
app.include_router(articles.router, prefix="/api/articles")
app.include_router(metadata.router, prefix="/api/metadata")
app.include_router(faqs.router, prefix="/api/faqs")
app.include_router(social.router, prefix="/api/social")


@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "SEO Content API funcionando correctamente. Abre /docs para probar los endpoints.",
    }
