from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


Platform = Literal["twitter", "linkedin", "instagram", "facebook"]


class SocialRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_title: str = Field(..., min_length=5, max_length=160)
    article_content: str = Field(..., min_length=100, max_length=30000)
    target_platforms: list[Platform] = Field(..., min_length=1, max_length=4)


class TwitterContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=20, max_length=280)
    hashtags: list[str] = Field(..., min_length=1, max_length=4)
    call_to_action: str = Field(..., min_length=5, max_length=80)


class LinkedInContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=100, max_length=3000)
    hashtags: list[str] = Field(..., min_length=3, max_length=8)
    call_to_action: str = Field(..., min_length=5, max_length=140)


class InstagramContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    caption: str = Field(..., min_length=80, max_length=2200)
    hashtags: list[str] = Field(..., min_length=5, max_length=15)
    call_to_action: str = Field(..., min_length=5, max_length=120)


class FacebookContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=80, max_length=2000)
    hashtags: list[str] = Field(..., min_length=1, max_length=6)
    call_to_action: str = Field(..., min_length=5, max_length=120)


class SocialResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    twitter: TwitterContent | None = None
    linkedin: LinkedInContent | None = None
    instagram: InstagramContent | None = None
    facebook: FacebookContent | None = None
