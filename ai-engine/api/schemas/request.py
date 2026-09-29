from pydantic import BaseModel, HttpUrl, Field, field_validator


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    title: str = Field(default="", max_length=500)
    content: str = Field(min_length=100, max_length=200_000)

    @field_validator("url")
    @classmethod
    def public_url(cls, value):
        from pipeline.security.public_http import validate_url
        validate_url(str(value), resolve=False)
        return value
