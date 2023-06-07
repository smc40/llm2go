from fastapi import FastAPI

from base import TextTransfer, LLMQuestion
from llm.base import LLM

app = FastAPI()


@app.get("/helloworld", response_model=TextTransfer)
async def helloworld():
    return {"text": "hello world"}


@app.post("/drugname", response_model=TextTransfer)
async def drugname(llm_question: LLMQuestion):
    model = llm_question.model
    llm = LLM(model=model)
    answer = llm.apply(text=llm_question.text)
    return {"text": answer}
