from transformers import T5Tokenizer, T5ForConditionalGeneration


class LLM:

    def __init__(self, model: str):
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

        self._prompt_template = "In the following sentence, what is the drug name: {text}"

    def apply(self, text: str):
        prompt = self._prompt_template.format(text=text)
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self._model.generate(input_ids, max_length=512)

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    model = "google/flan-t5-small"
    llm = LLM(model=model)

    response = llm.apply("Ibuprofen is well known to cause diarrhea.")
    print(response)
