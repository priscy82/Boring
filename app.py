from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

MODEL_NAME = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.get("/him")
def him(query: str):
    prompt = f"{query}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_p=0.95, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": result[len(prompt):].strip()}
