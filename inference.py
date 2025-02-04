import torch
from transformers import AutoTokenizer
from llama_modeling.front_end import LlamaForCausalLM
from llama_modeling.config import LlamaConfig
import json
import sys
from utils.trainutils import load_checkpoint

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_model>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("config.json") as f:
        config_dict = json.load(f)
    config = LlamaConfig(**{k: v for k, v in config_dict.items() if k in LlamaConfig.__dataclass_fields__})
    
    model = LlamaForCausalLM(config).to(device)
    
    load_checkpoint(model, model_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("./SmolLM2-135M-Instruct")
    
    prompts = [
        "Once upon a time,",
        "The best way to learn programming is",
        "Here's a recipe for chocolate cake:"
    ]
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=None):
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            output = generate_text(model, tokenizer, prompt)
            print(f"Generated: {output}")
            print("-" * 50)

if __name__ == "__main__":
    main()