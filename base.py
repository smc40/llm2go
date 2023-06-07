from pydantic import BaseModel


class TextTransfer(BaseModel):
    text: str


class IntTransfer(BaseModel):
    number: int
