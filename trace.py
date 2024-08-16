
import os
import sys
import time
import numpy as np
from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import (
    GPT2LMHeadModel,
)
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import (
    GPT2Tokenizer,
)

import np_gpt2 as np_gpt2

def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        func_name = code.co_name
        lineno = frame.f_lineno
        filename = code.co_filename
        print(f"Calling function: {func_name} at {filename}:{lineno}")
    return trace_calls

prompt = "Electric Light Orchestra"
model = "gpt2"
max_length = 15

gpt2 = GPT2LMHeadModel.from_pretrained(
    model,
    output_attentions=True,
    activation_function="gelu",
    attn_implementation="eager",
)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
    model, clean_up_tokenization_spaces=True
)
parameters = np_gpt2.get_parameters(gpt2)

gpt2_model = np_gpt2.NpGPT2(
    parameters, gpt2_tokenizer, decode_blocks=12, attn_heads=12, embedding_size=768
)


# Set the trace function
# sys.settrace(trace_calls)
np_gen_text = gpt2_model.generate(prompt, max_token_len=max_length)
# sys.settrace(None)
