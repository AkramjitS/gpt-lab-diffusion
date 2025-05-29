import os
import sys
import uuid
import time
import copy
import glob
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import argparse
import itertools
from itertools import islice
import tiktoken
import json
import datetime
import pickle
import shutil
import csv
import random
import math
import numpy as np # Import numpy for potential future use, set random seed now not to forget to set it later
from datasets import load_dataset
from gpt.helper import *
from gpt.model import *
from gpt.hellaswag import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
from torch.utils.data import DataLoader
#torch._inductor.config.max_autotune_gemm_backends = ["ATEN"]

@dataclass
class Hyperparameters:
    train_folder:str = 'experiments/20250529_172823_ModdedGPT'
    checkpoint_name:str = 'state_step035000.pt'
    additional_inputs:list[str] = field(default_factory=list)
    cuda:int = 1
    
    num_layers:int = 12
    num_val_emb:int = 2
    num_heads:int = 6
    model_dim:int = 384
    max_seq_len:int = 4096
    mlp_ratio:int = 4
    num_generate_steps:int = 50
    
    
    def __post_init__(self):
        self.additional_inputs = [
            "Once upon a time,",
            "The meaning of life is",
            "In the year 2026,",
            "I'm a Large Language Model (LLM), which means",
            "2 + 3 = "
        ] + self.additional_inputs
        return
    
    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Create continuations through generation for the supplied model for the supplied inputs")
        
        parser.add_argument('--train_folder', type=str, help='The folder where the model checkpoint exists')
        parser.add_argument('--checkpoint_name', type=str, help='The checkpoint name in the train_folder')
        
        parser.add_argument('--additional_inputs', type=str, default=[], nargs='*', help='The additional input to generate with')
        
        parser.add_argument('--cuda', type=int, help='The default cuda device to use')
        
        parser.add_argument('--num_layers', type=int, help='Number of layers in the network')
        parser.add_argument('--num_val_emb', type=int, help='Number of value embeddings in the network')
        parser.add_argument('--num_heads', type=int, help='Number of heads in multihead attention in the network')
        parser.add_argument('--model_dim', type=int, help='Number of dimensions for each token in the network')
        parser.add_argument('--max_seq_len', type=int, help='The max sequence length in the context')
        parser.add_argument('--mlp_ratio', type=int, help='Ratio to upscale and then downscale in transformer blocks in the network')
        parser.add_argument('--num_generate_steps', type=int, help='Max number of generation steps in diffusion')
        
        args = parser.parse_args()
        
        # Create a base instance with defaults
        instance = cls()
        
        # Update instance with command-line arguments that were provided
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                setattr(instance, key, value)
        
        # Run post_init validations after applying CLI arguments
        instance.__post_init__()
        
        return instance, args
    
def get_model_tokenizer_args():
    args, cli_args = Hyperparameters.from_args()
    
        # Check if environment variables are set by torchrun, otherwise default to single GPU
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        # Multi-GPU setup with torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single GPU setup
        rank = args.cuda
        world_size = 1
        local_rank = args.cuda
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    print(f"Running with {world_size} GPU{'s' if world_size > 1 else ''}")
    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
        # Initialize distributed process group if using multiple GPUs
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = (rank == args.cuda)  # this process will do logging, checkpointing etc.
    
    
    # Handle tokenizer separately
    try:
        from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, model_max_length=args.max_seq_len)
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, model_max_length=1024)
    except Exception as e:
        print(f"- Tokenizer distilroberta-base failed to load. Error: {e}")
        exit()
    
    model: nn.Module = Diffusion(vocab_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
        #eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        num_layers=args.num_layers,
        num_val_emb=args.num_val_emb,
        num_heads=args.num_heads, 
        model_dim=args.model_dim,
        max_seq_len=args.max_seq_len,
        mlp_ratio=args.mlp_ratio,
        num_steps=args.num_generate_steps).requires_grad_(False)
    
    checkpoint = torch.load(args.train_folder + '/' + args.checkpoint_name)
    model.load_state_dict(checkpoint['model'], strict = False, assign = True)
    model.to('cuda')
    model.eval()

    return model, tokenizer, args
    
def sample_from_model(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate text samples from the model given a prompt."""
    encode = tokenizer.encode
    decode = tokenizer.decode
        
    # Encode the prompt
    input_ids = encode(prompt)[:-1]
    x = torch.tensor(input_ids, dtype=torch.int32, device="cuda")

    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode and return
        return [decode(y.tolist()) for y in outputs]

def generate_for_prompts(model, tokenizer, args):
    # Then at the end of training:
    print("-"*10 + " EXAMPLE MODEL GENERATIONS AFTER TRAINING " + "-"*10)
    prompts = args.additional_inputs
    for prompt in prompts:
        continuations = sample_from_model(model, tokenizer, prompt, max_new_tokens=16)
        for index, continuation in enumerate(continuations):
            print(f"iteration: {index}, continuation={continuation}")
    
if __name__ == "__main__":
    model, tokenizer, args = get_model_tokenizer_args()
    generate_for_prompts(model, tokenizer, args)