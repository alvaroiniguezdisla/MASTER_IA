from pydantic import BaseModel, ConfigDict, Field


class FAQRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_content: str = Field(..., min_length=200, max_length=30000)
    max_questions: int = Field(default=5, ge=3, le=10)


class FAQ(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=10, max_length=160)
    answer: str = Field(..., min_length=50, max_length=900)


class FAQResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faqs: list[FAQ] = Field(..., min_length=3, max_length=10)
    json_ld_schema: dict = Field(..., description="Schema.org FAQPage en formato JSON-LD válido.")
