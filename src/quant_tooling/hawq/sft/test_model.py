from transformers import AutoTokenizer
from transformers import AutoModel
import torch

# model_name = "Salesforce/CoDA-v0-Instruct"
# model_name = "./checkpoints_coda/wikitext2_final"
model_name = "./checkpoints_coda/gsm8k_final"

print(f"Loading: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

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
