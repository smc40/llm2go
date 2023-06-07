# llm2go
**Goal:** API Service for interchangeable, locally hosted LLMs

**User Story:** As a *Data Scientist* I would like to *quickly use and test newly released LLMs* so *I can compare thir performances*.

## Links
- FastAPI: https://fastapi.tiangolo.com/
- LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- Inspiration or even solution?!: https://github.com/huggingface/text-generation-inference

## Use Huggingface Models
Install transformers
```
pip install transformers, torch, SentencePiece, accelerate
```
download and use a model
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

prompt = "In the following sentence, what is the drugname: Ibuprofen is well known to cause diarrhia."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length = 512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
