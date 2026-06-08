from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


HeadingLevel = Literal["H1", "H2", "H3"]


class ArticleSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading_level: HeadingLevel = Field(..., description="Nivel jerárquico SEO: H1, H2 o H3.")
    heading: str = Field(..., min_length=5, max_length=120)
    content: str = Field(..., min_length=50, max_length=4000)


class ArticleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_keyword: str = Field(..., min_length=2, max_length=120)
    secondary_keywords: list[str] = Field(default_factory=list, max_length=20)
    word_count: int = Field(default=900, ge=500, le=2500)
    tone: str = Field(default="profesional e informativo", min_length=3, max_length=80)


class ArticleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=10, max_length=120)
    sections: list[ArticleSection] = Field(..., min_length=4, max_length=20)
    keyword_density: dict[str, float] = Field(..., description="Densidad aproximada de keywords en porcentaje.")
    call_to_actions: list[str] = Field(..., min_length=2, max_length=5)
