from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd


class LLM:

    def __init__(self, model: str):
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._model = T5ForConditionalGeneration.from_pretrained(model)

        self._prompt_template = "In the following sentence, what is the drug name: {text}"

    def apply(self, text: str):
        prompt = self._prompt_template.format(text=text)
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self._model.generate(input_ids, max_length=512)

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_performance(model: str, filename: str) -> float:
    with open(Path.cwd().parent / filename) as tfile:
        df = pd.read_csv(tfile)

    llm = LLM(model=model)

    comp = []
    for _, (sent, drug) in df.iterrows():
        pred = llm.apply(text=sent)
        if pred.lower() != drug.lower():
            print(pred, drug, sent)


if __name__ == '__main__':
    model = "google/flan-t5-small"
    # llm = LLM(model=model)
    #
    # response = llm.apply("Ibuprofen is well known to cause diarrhea.")
    # print(response)
    compute_performance(model=model, filename='100sentences.csv')
