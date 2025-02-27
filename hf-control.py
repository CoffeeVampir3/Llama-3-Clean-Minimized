import torch
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = total_params * 2 / 1024 / 1024
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter size (MB): {param_size:.2f}")
    return total_params, trainable_params, param_size

mpath = "SmolLM2-135M-Instruct"

device="cuda"
forward_dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(f"./{mpath}/")
tokenizer = AutoTokenizer.from_pretrained(f"./{mpath}/")

hf_stats = count_parameters(model)

from transformers import AutoConfig
from pprint import pprint

config = AutoConfig.from_pretrained(f"./{mpath}/")
pprint(config.to_dict(), width=1)

messages = [{"role": "user", "content": "Hello! What is your name?"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")
imstart = tokenizer("<|im_start|>assistant\n", return_tensors="pt")
combined_input = torch.cat([inputs.input_ids, imstart.input_ids], dim=1)

# Does not like being compiled, never completes in my test. No idea why.
# start_time = time.time()
# model = torch.compile(model)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Compiling took: {elapsed_time:.2f} seconds")

start_time = time.time()
output = model(combined_input)
with torch.no_grad(), torch.amp.autocast(device, dtype=forward_dtype):
    for _ in range(30):
        output = model(combined_input)
        logits = output.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        combined_input = torch.cat([combined_input, next_token.unsqueeze(0)], dim=1)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Final output: {tokenizer.decode(combined_input[0])}")
print(f"Execution time: {elapsed_time:.2f} seconds")