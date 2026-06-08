from pydantic import BaseModel, ConfigDict, Field, model_validator


class MetadataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_title: str = Field(..., min_length=5, max_length=160)
    main_keyword: str = Field(..., min_length=2, max_length=120)
    article_excerpt: str = Field(..., min_length=20, max_length=1000)


class MetaTitle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=10, max_length=60)
    character_count: int = Field(..., ge=10, le=60)

    @model_validator(mode="after")
    def validate_real_length(self):
        if self.character_count != len(self.text):
            raise ValueError("character_count debe coincidir con la longitud real del meta title.")
        if len(self.text) > 60:
            raise ValueError("El meta title no puede superar 60 caracteres.")
        return self


class MetaDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=50, max_length=160)
    character_count: int = Field(..., ge=50, le=160)

    @model_validator(mode="after")
    def validate_real_length(self):
        if self.character_count != len(self.text):
            raise ValueError("character_count debe coincidir con la longitud real de la meta description.")
        if len(self.text) > 160:
            raise ValueError("La meta description no puede superar 160 caracteres.")
        return self


class MetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta_titles: list[MetaTitle] = Field(..., min_length=3, max_length=5)
    meta_descriptions: list[MetaDescription] = Field(..., min_length=3, max_length=5)
