import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
checkpoint = "HuggingFaceTB/SmolLM2-360M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for fp16 use `torch_dtype=torch.float16` instead
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

prompt1 = "Currently I am trying to run particle monte carlo simulation for deep reactive ion etching process (also known as bosch process), to calculate etch rates from the plasma Ion energy distribution functions, I must"
prompt2 = "Currently I am running particle monte carlo simulation for deep reactive ion etching process (also known as bosch process), to calculate etch rates from the plasma Ion energy distribution functions, I must"
inputs = tokenizer.encode(prompt1, return_tensors="pt").to("cpu")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
