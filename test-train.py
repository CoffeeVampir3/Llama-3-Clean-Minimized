import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math, os, sys, json, glob, time, random
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo
from cut_cross_entropy import linear_cross_entropy
from torch.nn.utils import clip_grad_norm_
from utils.trainutils import count_parameters_layerwise, save_checkpoint, TBLogger

from llama_modeling.front_end import LlamaForCausalLM
from llama_modeling.config import LlamaConfig

class JSONLDataset(Dataset):
    def __init__(self, directory_path, tokenizer, seq_length=1024, 
                 text_key="text", max_files=None, batch_size=1000, 
                 pad_token_id=0):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.sequences = []

        files = glob.glob(os.path.join(directory_path, "*.jsonl"))
        if max_files is not None:
            files = files[:max_files]

        text_batch = []
        for file_idx, file_path in enumerate(files):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get(text_key, "")
                        if len(text) >= 100:
                            text_batch.append(text)
                            
                            if len(text_batch) >= batch_size:
                                self._process_batch(text_batch)
                                text_batch = []
                    except:
                        continue
        
        if text_batch:
            self._process_batch(text_batch)

        if self.sequences:
            self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        else:
            self.sequences = torch.empty((0, seq_length), dtype=torch.long)

    def _process_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_tensors=None
        )['input_ids']

        mlen = 0
        for token_ids in encoded:
            for i in range(0, len(token_ids), self.seq_length):
                chunk = token_ids[i:i+self.seq_length]
                
                # Pad
                if len(chunk) < self.seq_length:
                    chunk += [self.pad_token_id] * (self.seq_length - len(chunk))
                
                self.sequences.append(chunk)
                mlen = max(mlen, len(chunk))
                
        print("MAX: ", mlen)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def train_model(model, train_loader, optimizer, device, epochs=5, forward_dtype=torch.float32):
    model.train()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    
    logger = TBLogger(log_dir=f'logs/run-{time.time()}')
    
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=5e-6 
    )
    
    model = torch.compile(
        model,
    )
    
    global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type='cuda', dtype=forward_dtype):
                hidden_states, classifier_weights = model(data)
                
                loss = linear_cross_entropy(
                    hidden_states,
                    classifier_weights,
                    data,
                    shift=True,
                    reduction="mean"
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update metrics - just add the loss itself
            running_loss += loss.item()
            total_batches += 1
            global_step += 1
            avg_loss = running_loss / total_batches
            perplexity = math.exp(min(avg_loss, 100))

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{perplexity:.2f}'
            })
            
            metrics = {
                'loss': loss.item(),
                'perplexity': perplexity,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'batch_size': data.size(0)
            }

            logger.log(metrics, step=global_step, model=model, grad_checking=True)

            if batch_idx % 100 == 0:
                print(f'\nBatch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {avg_loss:.4f}, '
                      f'Perplexity: {perplexity:.2f}, '
                      f'Batches Processed: {total_batches}')

        epoch_loss = running_loss / total_batches
        epoch_ppl = math.exp(min(epoch_loss, 100))
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Perplexity: {epoch_ppl:.2f}')
        print(f'Total Batches Processed: {total_batches}\n')
        
        save_checkpoint(model, f'epoch_{epoch+1}.safetensors')

def sample_examples(dataset, tokenizer, num_samples=5):
    if len(dataset) == 0:
        print("The dataset is empty.")
        return
    
    num_samples = min(num_samples, len(dataset))
    
    sampled_indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(sampled_indices):
        sequence = dataset[idx]
        print(f"Sample {i + 1} (Index {idx}):")
        print(sequence)
        decoded_text = tokenizer.decode(sequence, skip_special_tokens=False, decode_special_tokens=False)
        print(decoded_text)
        print("-" * 40)
      
def main():
    BATCH_SIZE = 72
    SEQ_LENGTH = 256
    EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("./SmolLM2-135M-Instruct")

    config_path = "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    config = LlamaConfig(**{k: v for k, v in config_dict.items() if k in LlamaConfig.__dataclass_fields__})

    model = LlamaForCausalLM(config).to("cuda")

    dataset = JSONLDataset(
        directory_path="./Data_big",
        tokenizer=tokenizer,
        seq_length=SEQ_LENGTH,
        text_key="text",
        max_files=None,
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True 
    )

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        epsilon=1e-12,
        weight_decay=1e-05,
        max_preconditioner_dim=2048,
        precondition_frequency=100,
        start_preconditioning_step=250,
        use_decoupled_weight_decay=False,
        grafting_config=AdamGraftingConfig(
            beta2=0.999,
            epsilon=1e-12,
        ),
    )
    
    print("*"*100)
    torch.set_float32_matmul_precision('high')
    
    count_parameters_layerwise(model)

    train_model(model, train_loader, optimizer, DEVICE, EPOCHS, forward_dtype=torch.bfloat16)

if __name__ == "__main__":
    main()