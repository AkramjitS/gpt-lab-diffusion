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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
#torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
from torch.utils.data import DataLoader
#torch._inductor.config.max_autotune_gemm_backends = ["ATEN"]

from transformers import T5Tokenizer, T5ForConditionalGeneration

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
    num_generate_steps:int = 1 # set to one as all steps act identically
    
    
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
        #parser.add_argument('--num_generate_steps', type=int, help='Max number of generation steps in diffusion')
        
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
    
    print(f"input cli_args: {cli_args}")
    
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
    
    model: nn.Module = Tracing_Diffusion(vocab_size=tokenizer.vocab_size,
        mask_token_id=tokenizer.mask_token_id,
        bos_token_id=tokenizer.bos_token_id,
        num_layers=args.num_layers,
        num_heads=args.num_heads, 
        model_dim=args.model_dim,
        max_seq_len=args.max_seq_len,
        mlp_ratio=args.mlp_ratio,
        num_steps=args.num_generate_steps).requires_grad_(False)
    
    checkpoint = torch.load(args.train_folder + '/' + args.checkpoint_name)
    model.load_state_dict(checkpoint['model'], strict = False, assign = True)
    model.to('cuda')
    model.eval()
    
    top_k_descriptions = checkpoint['top_k_descriptions']

    return model, top_k_descriptions, tokenizer, args
    
def sample_from_model(model, top_k_descriptions, tokenizer, desc_model, desc_tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate text samples from the model given a prompt."""
    encode = tokenizer.encode
    decode = tokenizer.decode
        
    # Encode the prompt
    input_ids = encode(prompt)[:-1]
    x = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
    
    def pre_prompt(num_inputs: int, weighted: bool, len_response_words: int) -> str:
        weighted_string = "weighted" if weighted else ""
        return (
            f"Identify the single unifying theme among these {num_inputs} {weighted_string} fragments.\n"
            "- Do not copy or quote any phrase verbatim—use your own wording.\n"
            "- Your answer must be a genuine synthesis (not a substring of any input).\n"
            f"- Use no more than {len_response_words} words total.\n"
            "- Output exactly in this format: response=<your answer>\n"
            "\nFragments:\n"
        )
    def post_prompt() -> str:
        return (
            "\nresponse="
        )

    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        model_outputs = []
        activations = []
        for generation_index, generation in enumerate(outputs[1:], 1):
            generation_output, generation_activation = generation
            generation_model_output = decode(generation_output.tolist())
            model_outputs.append(generation_model_output)
            generation_activation = [layer_act.mean(dim=(0, 1)) for layer_act in generation_activation]
            layer_activation_descriptions = []
            for layer_index in range(len(generation_activation)):
                layer_description = []
                total_activation = generation_activation[layer_index].sum()
                for position, pos_description in top_k_descriptions[layer_index].items():
                    # we assume at least one position doesn't have an empty description
                    if pos_description == '':
                        continue
                    importance = generation_activation[layer_index][position] / total_activation
                    layer_description.append((importance.item(), pos_description))
                    layer_description.sort(key=lambda pair: pair[0], reverse=True)
                
                if len(layer_description) > 10:
                    layer_description = layer_description[:10]
    
                intermediate_prompt = ""
                relative_total_importance = sum(pos[0] for pos in layer_description)
                for index, layer_description_pair in enumerate(layer_description, 1):
                    pos_importance, pos_desc = layer_description_pair
                    pos_importance /= relative_total_importance
                    intermediate_prompt += f"{index}. weight={pos_importance:.3f} input=\"{pos_desc}\"\n"
                complete_prompt = pre_prompt(len(layer_description), True, 20) + intermediate_prompt + post_prompt()
                
                t5_input = desc_tokenizer(complete_prompt).input_ids
                t5_output = desc_model.generate(torch.tensor([t5_input], device='cuda'))
                t5_output_plaintext = desc_tokenizer.decode(t5_output[0], skip_special_tokens=True)
                layer_activation_descriptions.append(t5_output_plaintext)
            
            generation_intermediate_prompt = ""
            for index, layer_description in enumerate(layer_activation_descriptions, 1):
                generation_intermediate_prompt += f"{index} input=\"{layer_description}\"\n"
            generation_complete_prompt = pre_prompt(len(layer_activation_descriptions), False, 50) + generation_intermediate_prompt + post_prompt()
            
            generation_t5_input = desc_tokenizer(generation_complete_prompt).input_ids
            generation_t5_output = desc_model.generate(torch.tensor([generation_t5_input], device='cuda'))
            generation_t5_output_plaintext = desc_tokenizer.decode(generation_t5_output[0], skip_special_tokens=True)
            
            evaluation_output_prompt = (
                "Given the following prompt="
                f"\"{prompt}\"\n"
                "Our model produced the completion="
                f"\"{generation_model_output}\"\n"
                "Score the models completion on a scale from 1 to 10.\n"
                "response="
            )
            
            evaluation_output_t5_input = desc_tokenizer(evaluation_output_prompt).input_ids
            evaluation_output_t5_output = desc_model.generate(torch.tensor([evaluation_output_t5_input], device='cuda'))
            evaluation_output_t5_output_plaintext = desc_tokenizer.decode(evaluation_output_t5_output[0], skip_special_tokens=True)
            
            evaluation_mechanistic_interpretability_pre_prompt = (
                "Given the following prompt="
                f"\"{prompt}\"\n"
                "Our model produces the following completion="
                f"\"{generation_model_output}\"\n"
                "We are experimenting with automatic interpretability and thus have the following interpretations"
                " for each block in the model for how the completion was generated.\n"
            )
            evaluation_mechanistic_interpretability_post_prompt = (
                "The following was an automatic summary of the interpretation of the models generation process using the per layer interpretations.\n"
                f"\"{generation_t5_output_plaintext}\"\n"
                "Score the models automatic mechanistic interpretability on a scale from 1 to 10.\n"
                "response="
            )
            evaluation_mechanistic_interpretability_intermediate_prompt = ""
            for layer_index, layer_description in enumerate(layer_activation_descriptions, 1):
                evaluation_mechanistic_interpretability_intermediate_prompt += f"layer={layer_index}, interpretation=\"{layer_description}\"\n"
                
            evaluation_mechanistic_interpretability_total_prompt = evaluation_mechanistic_interpretability_pre_prompt + evaluation_mechanistic_interpretability_intermediate_prompt + evaluation_mechanistic_interpretability_post_prompt
            
            evaluation_mechanistic_interpretability_t5_input = desc_tokenizer(evaluation_mechanistic_interpretability_total_prompt).input_ids
            evaluation_mechanistic_interpretability_t5_output = desc_model.generate(torch.tensor([evaluation_mechanistic_interpretability_t5_input], device='cuda'))
            evaluation_mechanistic_interpretability_t5_output_plaintext = desc_tokenizer.decode(evaluation_mechanistic_interpretability_t5_output[0], skip_special_tokens=True)
            
            activations.append((layer_activation_descriptions, generation_t5_output_plaintext, evaluation_output_t5_output_plaintext, evaluation_mechanistic_interpretability_t5_output_plaintext))
            
        return model_outputs, activations
        
        # Decode and return
        #return [decode(y.tolist()) for y in outputs]

def generate_for_prompts(model, top_k_descriptions, tokenizer, args):
    desc_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl", model_max_length=args.max_seq_len)
    desc_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", load_in_8bit=True)
    # already loaded into cuda and applying .cuda() throws an error. 
    #desc_model = desc_model.cuda()
    desc_model = desc_model.eval()
    
    # Then at the end of training:
    print("-"*10 + " EXAMPLE MODEL GENERATIONS AFTER TRAINING " + "-"*10)
    prompts = args.additional_inputs
    for prompt in prompts:
        generations = sample_from_model(model, top_k_descriptions, tokenizer, desc_model, desc_tokenizer, prompt, max_new_tokens=16)
        print(f"input: {prompt}")
        model_outputs, activations = generations
        for index in range(len(activations)):
            layer_descriptions, total_description, eval_model, eval_mech_interp = activations[index]
            print_string = (
                f"iteration: {index + 1}\n\t"
                f"continuation={model_outputs[index]}\n\t"
                f"per layer description={layer_descriptions}\n\t"
                f"total description={total_description}\n\t"
                f"models output score={eval_model}\n\t"
                f"models mechanistic interpretability score={eval_mech_interp}"
            )
            print(print_string)
    
if __name__ == "__main__":
    model, top_k_descriptions, tokenizer, args = get_model_tokenizer_args()
    generate_for_prompts(model, top_k_descriptions, tokenizer, args)