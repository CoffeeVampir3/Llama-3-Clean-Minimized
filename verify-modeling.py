import math, random, json, time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.cuda.amp import autocast
from safetensors import safe_open
from transformers import AutoTokenizer

from llama_modeling.front_end import LlamaForCausalLM
from llama_modeling.config import LlamaConfig

# https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = total_params * 2 / 1024 / 1024
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter size (MB): {param_size:.2f}")
    return total_params, trainable_params, param_size

def load_model(model_path: str, config_path: str, device):
    with open(config_path) as f:
        config_dict = json.load(f)
    config = LlamaConfig(**{k: v for k, v in config_dict.items() if k in LlamaConfig.__dataclass_fields__})

    from pprint import pprint
    pprint(config, width=1)
    
    model = LlamaForCausalLM(config)
    
    with safe_open(model_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(state_dict, strict=False)
    
    return model.to(device)

device="cuda"
forward_dtype = torch.float16
model = load_model("./SmolLM2-135M-Instruct/model.safetensors", "./SmolLM2-135M-Instruct/config.json", device)

count_parameters(model)

tokenizer = AutoTokenizer.from_pretrained("./SmolLM2-135M-Instruct")

messages = [{"role": "user", "content": "Hello! What is your name?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")
imstart = tokenizer("<|im_start|>assistant\n", return_tensors="pt")
combined_input = torch.cat([inputs.input_ids, imstart.input_ids], dim=1).to(device)

# Verify we can compile the model.
start_time = time.time()
model = torch.compile(model)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Compiling took: {elapsed_time:.2f} seconds")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()
with torch.no_grad(), torch.amp.autocast(device, dtype=forward_dtype):
    output_ids = model.generate(combined_input.to(device), max_new_tokens=30, temperature=0.0)
end_time = time.time()
elapsed_time = end_time - start_time

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False, decode_special_tokens=False)
print(generated_text)
print(f"Execution time: {elapsed_time:.2f} seconds")