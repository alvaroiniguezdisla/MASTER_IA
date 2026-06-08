from app.services.openai_client import AzureOpenAIService
from app.services.keywords_service import KeywordService
from app.services.articles_service import ArticleService
from app.services.metadata_service import MetadataService
from app.services.faqs_service import FAQService
from app.services.social_service import SocialService


_openai_service = AzureOpenAIService()


def get_keyword_service() -> KeywordService:
    return KeywordService(_openai_service)


def get_article_service() -> ArticleService:
    return ArticleService(_openai_service)


def get_metadata_service() -> MetadataService:
    return MetadataService(_openai_service)


def get_faq_service() -> FAQService:
    return FAQService(_openai_service)


def get_social_service() -> SocialService:
    return SocialService(_openai_service)
