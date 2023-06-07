from pydantic import BaseModel


class Text(BaseModel):
    text: str

class Performance(BaseModel):
    performance: float


class LLMQuestion(BaseModel):
    model_name: str
    text: str
