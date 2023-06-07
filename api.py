from fastapi import FastAPI

from base import TextTransfer, IntTransfer

app = FastAPI()


@app.get("/helloworld", response_model=TextTransfer)
async def helloworld():
    return {"text": "hello world"}

@app.post("/detect", response_model=TextTransfer)
async def printnumber(sentence: TextTransfer):
    original_number = inputdata.number
    new_number = original_number*2
    return {"text": f'original number: {original_number} / new number: {new_number}'}
