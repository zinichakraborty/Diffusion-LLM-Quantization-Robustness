from transformers import AutoTokenizer, AutoModel
import torch

model_name = "Salesforce/CoDA-v0-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load with AutoModel to get the base model
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="cuda", torch_dtype=torch.bfloat16)

prompt = "Write a Python function to calculate fibonacci numbers"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Check if model has a custom inference method
outputs = None
if hasattr(model, 'diffusion_generate'):
    print("===== generating with diffusion_generate =====")
    outputs = model.diffusion_generate(
        inputs.input_ids,
        num_steps=4,
        max_length=256
    )
elif hasattr(model, 'sample'):
    print("===== generating with model sample =====")
    outputs = model.sample(
        inputs.input_ids,
        steps=128,
        max_tokens=256
    )
else:
    print("===== Model methods:", [m for m in dir(model) if not m.startswith('_')])

print("===== Generation: ")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

