import numpy as np
import time
from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
)
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import (
    GPT2Tokenizer,
)

# GPT2 - 12 blocks, 12 heads, emb_size 768
# GPT2-med - 24 blocks, 16 heads, emb_size 1024
# GPT2-large - 36 blocks, 20 heads, emb_size 1280
# GPT2-xl - 48 blocks, 24 heads, emb_size 1600

prompt = "Electric Light Orchestra"
model = "gpt2"
max_length = 15

# Out of box Transformer package GPT2
import np_gpt2 as np_gpt2

print("Out of box Transformer")
print(
    np_gpt2.transformer_gpt2_inference(
        prompt, *np_gpt2.transformer_gpt2_model(model), max_length=max_length
    )
)
print('\n')


print("Edited Transformer")
# # Edited Transformer Package GPT2
t_gpt2 = GPT2LMHeadModel.from_pretrained(
    model,
    output_attentions=True,
    activation_function="gelu",
    attn_implementation="eager",
)  # loading gpt2 from forked hf_transformer library
t_gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
    model, clean_up_tokenization_spaces=True
)  # loading gpt2 tokenizer from forked hf_transformer library

inputs = t_gpt2_tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids

t_start_time = time.time()
gen_tokens = t_gpt2.generate(
    input_ids,
    do_sample=False,
    max_length=max_length,
    attention_mask=inputs["attention_mask"],
    pad_token_id=t_gpt2_tokenizer.pad_token_id,
)

t_gen_text = t_gpt2_tokenizer.batch_decode(gen_tokens)[0]
t_end_time = time.time()
print(t_gen_text)
print('\n')

# import model_trace as trace


print("np_gpt2")
# Custom GPT2
gpt2 = GPT2LMHeadModel.from_pretrained(
    model,
    output_attentions=True,
    activation_function="gelu",
    attn_implementation="eager",
)  # loading gpt2 from forked hf_transformer library
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
    model, clean_up_tokenization_spaces=True
)  # loading gpt2 tokenizer from forked hf_transformer library
parameters = np_gpt2.get_parameters(gpt2)

gpt2_model = np_gpt2.NpGPT2(
    parameters, gpt2_tokenizer, decode_blocks=12, attn_heads=12, embedding_size=768
)

np_start_time = time.time()
np_gen_text = gpt2_model.generate(prompt, max_token_len=max_length)
np_end_time = time.time()

print(f"np_GPT2:{np_end_time - np_start_time} \n{np_gen_text}")
