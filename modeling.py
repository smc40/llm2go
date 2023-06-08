from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import pandas as pd


class LLMTextGenerator(ABC):
    """Based on an input prompt the defined model generates a chunk of text

    Arguments:
        model_name (str): Name of the model.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._extract_drug_names_prompt = 'Extract all the drug names of the following text: {text}'
        self._summarize_prompt = 'Summarize the following text in {number_of_words} words: {text}'

    @abstractmethod
    def extract_drug_names(self, text: str) -> str:
        pass

    @abstractmethod
    def summarize(self, text: str, number_of_words: int) -> str:
        pass


class FlanT5Small(LLMTextGenerator):
    """Text generator based on the flan-t5-small model.

    Arguments:
        model_name (str): Name of the model.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self._model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self._model_name)

    def extract_drug_names(self, text: str) -> str:
        prompt = self._extract_drug_names_prompt.format(text=text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text: str, number_of_words: int) -> str:
        prompt = self._summarize_prompt.format(text=text, number_of_words=number_of_words)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



class CerebrasGPT(LLMTextGenerator):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)

    def extract_drug_names(self, text: str):
        prompt = self._extract_drug_names_prompt.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(**inputs, num_beams=5,
                                       max_new_tokens=50, early_stopping=True,
                                       no_repeat_ngram_size=2)
        text_output = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return text_output[0]

    def summarize(self, text: str, number_of_words: int) -> str:
        pass


class LLMTextGeneratorFactory:
    @staticmethod
    def get(model_name: str) -> LLMTextGenerator:
        match model_name:
            case 'google/flan-t5-small':
                return FlanT5Small(model_name=model_name)
            case 'cerebras/Cerebras-GPT-2.7B':
                return CerebrasGPT(model_name=model_name)

            # case 'bigscience/bloom-1b7':
            #     pass

            case _:
                raise ValueError(f'Unknown model {model_name}')


def compute_performance(model_name: str, filename: str) -> float:
    with open(Path.cwd() / filename) as tfile:
        df = pd.read_csv(tfile)

    llm = LLMTextGeneratorFactory().get(model_name=model_name)
    comp = []
    for _, (sent, drug) in tqdm(df.iterrows()):
        pred = llm.extract_drug_names(text=sent)
        comp.append(pred.lower() == drug.lower())

    return sum(comp) / df.shape[0]


if __name__ == '__main__':
    # model_name = "google/flan-t5-small"
    model_name = "cerebras/Cerebras-GPT-2.7B"
    accuracy = compute_performance(model_name=model_name, filename='100sentences.csv')
    print(f'Model {model_name} had accuracy {accuracy*100:.1f}%')
