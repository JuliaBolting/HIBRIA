from pydantic import BaseModel, HttpUrl, Field


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    title: str = Field(default="", max_length=500)
    content: str = Field(min_length=100, max_length=200_000)
