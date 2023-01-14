import torch
from transformers import GPT2Tokenizer, OPTForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

model = OPTForCausalLM.from_pretrained("facebook/opt-125m").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = (inputs.input_ids).to(device)

# Generate
generate_ids = model.generate(input_ids, max_length=30)
out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(out)