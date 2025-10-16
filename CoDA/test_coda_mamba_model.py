# IMPORTANT: total length = prompt_len + max_new_tokens
# `diffusion_generate` will pad to `max_length` internally.

# 6) Run inference
with torch.no_grad():
    out = model.diffusion_generate(
        inputs=input_ids,
        generation_config=gen_cfg,
        attention_mask=attn_mask,   # optional; pass if you batch-pad
    )

# 7) Detokenize
sequences = out.sequences if hasattr(out, "sequences") else out   # supports both return types
text = tok.batch_decode(sequences, skip_special_tokens=True)[0]
print(text)

