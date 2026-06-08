from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


IntentType = Literal["informacional", "transaccional"]


class KeywordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., min_length=3, max_length=120, description="Tema principal del contenido SEO.")
    industry: str = Field(..., min_length=2, max_length=100, description="Sector o industria.")
    language: str = Field(default="es", min_length=2, max_length=20, description="Idioma de la respuesta.")


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(..., min_length=2, max_length=140)
    intent: IntentType
    reason: str = Field(..., min_length=10, max_length=300)


class KeywordResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_keywords: list[str] = Field(..., min_length=5, max_length=15)
    long_tail_keywords: list[str] = Field(..., min_length=5, max_length=15)
    questions: list[str] = Field(..., min_length=5, max_length=15)
    intent_classification: list[IntentClassification] = Field(..., min_length=5, max_length=20)
