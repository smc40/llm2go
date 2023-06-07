




from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

class LLMTextGenerator:
    """Based on an input promt the defined model generates a chunck of text

    Attributes:
        model: HuggingFace hosted LLM model 
        device: String that indicates the detected device CPU or GPU.
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        """Create model for inference

        Args:
            tbd

        Raises:
            ValueError: If model is not present on huggingface

        """
        self.model_name = model_name
        self.extract_drugnames_prompt = 'Extract all the drug names of the following text: {text}'
        self.summarize_prompt = 'Summarize the following text in {number_of_words} words: {text}'

        if model_name == "google/flan-t5-small":
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    # def infer_model(self, text: str, number_of_words) -> str:
    #     prompt = self.prompt_template.format(text=text)
    #     input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
    #     outputs = self.model.generate(input_ids, max_length = 512)
    #     return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_drugnames(self, text: str) -> str:
        prompt = self.extract_drugnames_prompt.format(text=text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length = 512)
        if self.model_name == "google/flan-t5-small":
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            return  self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text: str, number_of_words: int) -> str:
        prompt = self.summarize_prompt.format(text=text, number_of_words=number_of_words)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length = 512)
        if self.model_name == "google/flan-t5-small":
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            pass