from pydantic import BaseModel


class TextTransfer(BaseModel):
    text: str


class LLMQuestion(BaseModel):
    model: str
    text: str
