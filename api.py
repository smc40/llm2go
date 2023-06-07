from fastapi import FastAPI

from base import Text, LLMQuestion, Performance
from modeling import LLMTextGeneratorFactory
from modeling import compute_performance

app = FastAPI()


@app.get("/helloworld", response_model=Text)
async def helloworld():
    return {"text": "hello world"}


@app.post("/drugname", response_model=Text)
async def drugname(data: LLMQuestion):
    llm = LLMTextGeneratorFactory().get(model_name=data.model_name)
    answer = llm.extract_drug_names(text=data.text)
    return {"text": answer}


@app.post("/performance", response_model=Performance)
async def performance(data: Text):
    model_name = data.text
    filename = '100sentences.csv'
    perf = compute_performance(model_name=model_name, filename=filename)
    return {"performance": perf}

