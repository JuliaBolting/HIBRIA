from pydantic import BaseModel, HttpUrl, Field


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    title: str = Field(default="")
    content: str = Field(min_length=1)