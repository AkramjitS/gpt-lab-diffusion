import os
import sys
import uuid
import time
import copy
import glob
from dataclasses import dataclass
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

@dataclass
class Hyperparameters:
    """
    default values are set to fit on a 2x GPUs w/ 8GB of VRAM each, but are not necessarily optimal
    """
    train_folder:str = 'experiments/20250529_172823_ModdedGPT'
    checkpoint_name:str = 'state_step035000.pt'
    model_name = "TopKGPT"
    # data
    #train_files = "data/fineweb*_train_*.bin" # input .bin to train on
    #val_files = "data/fineweb*_val_*.bin" # input .bin to eval validation loss on
    train_seq_len = 4*1024 # FlexAttention sequence length
    # optimization loop
    val_steps = 10 # number of steps to run validation for
    train_steps = 20#_000 # number of training steps to run
    grad_acc_steps = 1 # number of gradient accumulation steps per training step
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    # model size - parameters set for GPUs w/ 8GB VRAM
    num_layers = 12  # number of reansformer blocks
    num_heads = 6   # number of attention heads
    model_dim = 384  # size of model embedding vectors
    head_dim = None  # size of attention heads; if None, will default to model_dim // num_heads
    mlp_ratio = 4  # MLP hidden dimension is model_dim * mlp_ratio
    # memory optimization 
    use_fp8 = False # experimental; True on H100s (and newer?) should improve performance but seems to use more vram somehow
    # evaluation and logging
    val_loss_every = 100 # every how many steps to evaluate val loss? 0 for only at the end
    save_model = False
    # reproducibility
    seed: int | None = None # Optional random seed for initialization control

    # my variables
    cuda: int = 1
    num_generate_steps: int = 50
    train_val_dataset: str = "HuggingFaceFW/fineweb-edu"
    save_every: int | None = None

    # top_k specific variable
    top_k: int = 50

    def __post_init__(self):
        # Validate and set derived parameters
        assert self.train_seq_len % 128 == 0, f"train_seq_len must be multiple of 128, got {self.train_seq_len}"
        assert self.grad_acc_steps >= 1, f"grad_acc steps must be int >= 1"
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
        assert self.head_dim in [2 ** i for i in range(1, 10)], f"head_dim must be a power of 2, got {self.head_dim}"
        assert self.mlp_ratio > 0, f"mlp_ratio must be positive, got {self.mlp_ratio}"
        assert self.num_layers % 2 == 0, f"Number of layers ({self.num_layers}) must be even for skip connections"
        if self.save_every is not None:
            assert self.save_every > 0, f"save_every={self.save_every} must be a positive int"
            assert self.save_every % self.val_loss_every == 0, f"save_every={self.save_every} needs to be a multiple of val_loss_every={self.val_loss_every}"

    @classmethod
    def from_args(cls):
        """Create Hyperparameters from command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a GPT model with customizable hyperparameters")
        
        parser.add_argument('--train_folder', type=str, help='The folder where the model checkpoint exists')
        parser.add_argument('--checkpoint_name', type=str, help='The checkpoint name in the train_folder')
        
        # Data arguments
        parser.add_argument('--train_seq_len', type=int, help='Training sequence length')
        
        # Optimization arguments
        parser.add_argument('--val_steps', type=int, help='Number of steps to run validation for')
        parser.add_argument('--train_steps', type=int, help='Number of training iterations')
        parser.add_argument('--grad_acc_steps', type=int, help='Number of gradient accumulation steps per training iteration')
        parser.add_argument('--cooldown_frac', type=float, help='Fraction of training for learning rate cooldown')
        
        # Architecture arguments
        #parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
        parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, help='Number of attention heads')
        parser.add_argument('--model_dim', type=int, help='Model embedding dimension')
        parser.add_argument('--head_dim', type=int, help='Dimension per attention head')
        parser.add_argument('--mlp_ratio', type=int, help='MLP hidden dim ratio')
        
        # Other options
        parser.add_argument('--use_fp8', type=lambda x: (str(x).lower() == 'true'), default=None, 
                            help='experimental; True on H100s (and newer?) should improve performance but seems to use more vram somehow')
        parser.add_argument('--val_loss_every', type=int, help='Evaluate validation loss every N steps')
        parser.add_argument('--save_model', type=lambda x: (str(x).lower() == 'true'), default=None, help='Save model checkpoints')
        parser.add_argument('--model_name', type=str, help='Model name for logging')
        parser.add_argument('--seed', type=int, help='Random seed for initialization control')
        
        # my options
        parser.add_argument('--cuda', type=int, default=0,
                            help='provide which gpu to use')
        parser.add_argument('--num_generate_steps', type=int,
                            help='Maximum generatation steps')
        parser.add_argument('--train_val_dataset', type=str,
                            help='Dataset to load from HuggingFace for train and validation')
        parser.add_argument('--save_every', type=int,
                            help="save_every must be a postive int and a multiple of val_loss_every")
        parser.add_argument('--top_k', type=int,
                            help="how many values in the transcoders to record")
        
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

def main():
    # -----------------------------------------------------------------------------
    # int main

    # Parse arguments and create Hyperparameters instance
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

    #################################################
    #########           logging           ###########
    #################################################

    # begin logging
    logfile = None
    experiment_dir_path = None # Define experiment_dir_path outside the if block
    metrics_csv_path = None # Define metrics_csv_path
    if master_process:
        start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 1. Create the experiment directory name
        experiment_dir_name = (f"{start_time}_{args.model_name}")
        # 2. Create the experiment directory path
        experiment_dir_path = Path("experiments") / experiment_dir_name
        os.makedirs(experiment_dir_path, exist_ok=True)
        # 3. Set the logfile path inside the experiment directory
        logfile = experiment_dir_path / "training_log.txt"
        # 4. Set the metrics CSV file path
        metrics_csv_path = experiment_dir_path / "metrics.csv"
        print0(master_process, logfile, f"Logging to: {logfile}", console=True)
        print0(master_process, logfile, f"Metrics CSV: {metrics_csv_path}", console=True)
        # 5. Initialize metrics CSV file with headers
        with open(metrics_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step", "type", "loss", "cumulative_time_ms", "step_avg_ms"])
        # 6. Log any command-line arguments that were provided (overriding defaults)
        cli_arg_dict = {k: v for k, v in vars(cli_args).items() if v is not None}
        if cli_arg_dict:
            print0(master_process, logfile, "Command-line arguments overriding defaults:", console=True)
            for key, value in cli_arg_dict.items():
                print0(master_process, logfile, f"  --{key} = {value}", console=True)
            print0(master_process, logfile, "="*100, console=True)

        print0(master_process, logfile, "Copying relevant files to experiment directory...")
        files_to_copy = ["requirements.txt", sys.argv[0], "gpt/helper.py", "gpt/model.py"]
        for file_path_str in files_to_copy:
            file_path = Path(file_path_str)
            if file_path.exists():
                try:
                    # Use Path object methods for cleaner path manipulation
                    target_path = experiment_dir_path / f"{file_path.stem}.txt"
                    shutil.copy(str(file_path), str(target_path))
                    print0(master_process, logfile, f"- Copied {file_path} to {target_path}")
                except Exception as e:
                    print0(master_process, logfile, f"- Failed to copy {file_path}: {e}")
            else:
                print0(master_process, logfile, f"- File not found, skipping: {file_path}")
               
        # Handle tokenizer separately
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True, model_max_length=args.train_seq_len)
        except Exception as e:
            print0(master_process, logfile, f"- Tokenizer distilroberta-base failed to load. Error: {e}")
              
        def preprocess(example):
            #ids = tokenizer(example["text"], add_special_tokens=False).input_ids
            #return {"input_ids": ids + [tokenizer.eos_token_id]}
            ids = tokenizer(example["text"], add_special_tokens=True).input_ids
            return {"input_ids": ids + [tokenizer.pad_token_id] * random.randint(0, 10)} 
        # fineweb for train and validation
        try:
            ds = load_dataset(args.train_val_dataset, split="train", streaming=True)
            ds = ds.shuffle()
            ds = ds.map(preprocess, batched=False)

            all_chunks = ContinuousChunks(ds, block_size=args.train_seq_len)
            #val_slice   = islice(all_chunks, args.val_steps)
            #train_slice = islice(all_chunks, args.val_steps, None)
            #val_chunks   = SlicedChunks(val_slice)
            #train_chunks = SlicedChunks(train_slice)
            #train_loader = DataLoader(train_chunks, batch_size=1)
            #val_loader   = DataLoader(val_chunks,   batch_size=1)
            #train_loader = make_val_loader(all_chunks, ds, args.train_seq_len, args.val_steps, True)
            #val_loader   = make_val_loader(all_chunks, ds, args.train_seq_len, args.val_steps, False)
        except Exception as e:
            print0(master_process, logfile, f"- Error loading dataset: {args.train_val_dataset}. Error: {e}")
        # hellaswag dataset for test
        #try:
        #    ds_hellaswag = load_dataset("hellaswag", split="validation", streaming=True)
        #    ds_hellaswag = ds_hellaswag.map(preprocess, batched=False)
        #    
        #    hellaswag_chunks = ContinuousChunks
        #except Exception as e:
        #    print0(master_process, logfile, f"- Error loading dataset: hellaswag. Error: {e}")
        del preprocess

        print0(master_process, logfile, "="*100)

    # log information about the hardware/software environment this is running on
    print0(master_process, logfile, f"Running Python {sys.version}")
    print0(master_process, logfile, f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(master_process, logfile, nvidia_smi())
    print0(master_process, logfile, "="*100)

    #################################################
    #########      Seed for Reproducibility     #####
    #################################################

    # Set the seed *before* initializing the model or optimizer
    if args.seed is not None:
        print0(master_process, logfile, f"Setting random seed to {args.seed} for model initialization", console=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed) # Important for multi-GPU consistency
            # The following might be needed for full determinism, but can impact performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    ########################################
    #    Construct model                   #
    ########################################

    model: nn.Module = Tracing_Diffusion(vocab_size=tokenizer.vocab_size,
                        mask_token_id=tokenizer.mask_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads, 
                        model_dim=args.model_dim,
                        max_seq_len=args.train_seq_len,
                        mlp_ratio=args.mlp_ratio,
                        num_steps=args.num_generate_steps)
    print0(master_process, logfile, f'{model.get_num_params()} parameters', console=True)
    print0(master_process, logfile, model)

    # Set FP8 option based on hyperparameters
    #model.lm_head.use_fp8 = args.use_fp8

    for m in model.modules():
        if isinstance(m, nn.Embedding) or isinstance(m, Transcoder):
            m.bfloat16()
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)

    checkpoint = torch.load(args.train_folder + '/' + args.checkpoint_name)
    checkpoint = checkpoint['model']
    # the old model was saved compiled so we need to extract it according to the format expected in the model here, without the compiled name
    checkpoint = {k.split('.', 1)[1] : v for k, v in checkpoint.items()}

    #for k, v in model.state_dict().items():
    for k, v in model.named_parameters():
        v.data.copy_(checkpoint[k].data)
        v.requires_grad_(False)
    model = model.cuda()

    # Add fallback mode to handle compilation errors
    from torch import _dynamo
    torch._dynamo.config.suppress_errors = True
    
    # Use a more memory-efficient compilation option
    if args.use_fp8:
        model: nn.Module = torch.compile(model, dynamic=False)
    else:
        model: nn.Module = torch.compile(model, dynamic=False, mode="reduce-overhead")
        
    model.zero_grad(set_to_none=True)

    model = model.eval()

    ########################################
    #            Warmup kernels            #
    ########################################

    print0(master_process, logfile, "warming up kernels...", console=True)

    # Attempt to limit memory fragmentation
    if hasattr(torch.cuda, 'memory_stats'):
        print0(master_process, logfile, f"Initial GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

    print0(master_process, logfile, "kernels are toasty", console=True)

    ########################################
    #        Training and validation       #
    ########################################
    train_loader = make_val_loader(all_chunks, ds, args.train_seq_len, args.val_steps, True)
    train_iter = iter(train_loader)

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    # begin training
    cumulative_activations = None
    for step in range(args.train_steps + 1):
        last_step = (step == args.train_steps)

        # --------------- CHECKPOINT SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if last_step or (step != 0 and args.save_every is not None and step % args.save_every == 0): # inside validation section to avoid the if check every training iteration
                # 5. Save model checkpoint inside the experiment directory
                if master_process and args.save_model and experiment_dir_path:
                    assert cumulative_activations is not None, "cumulative_activations cannot be None. Run for atleast one iteration"
                    top_k_activations = [torch.topk(activation, args.top_k).indices for activation in cumulative_activations]

                    checkpoint = torch.load(args.train_folder + '/' + args.checkpoint_name)
                    log = dict(step=step, model=model.state_dict(), optimizers=checkpoint['optimizers'], top_k_activations=top_k_activations)
                    # Ensure experiment_dir_path exists (though it should from earlier)
                    os.makedirs(experiment_dir_path, exist_ok=True)
                    save_path = experiment_dir_path / f"state_step{step:06d}.pt"
                    torch.save(log, str(save_path))
                    print0(master_process, logfile, f"Saved checkpoint to {save_path}", console=True)
                # the last step only has the validation loop, so break to avoid training
                if last_step:
                    break

        # --------------- TRAINING SECTION -----------------
        loss = torch.tensor([0.], device="cuda")
        for _ in range(args.grad_acc_steps):
            inputs = torch.tensor(next(train_iter)['input_ids'], device='cuda')
            B = inputs.size(0)
            # Check if inputs exceed sequence length - can happen if the dataset has different sized examples
            if B > args.train_seq_len:
                inputs = inputs[:args.train_seq_len]
            t = torch.randint(0, model.num_steps, (1,), device='cuda')
            
            torch.compiler.cudagraph_mark_step_begin()
            logits, step_loss, transcoder_activations = model(
                inputs,
                t,
                inputs
            )
            loss += step_loss / args.grad_acc_steps
            transcoder_activations = [activation.mean(dim=(0, 1)) / args.grad_acc_steps for activation in transcoder_activations]
            if cumulative_activations is None:
                cumulative_activations = transcoder_activations
            else:
                for cumulative, current in zip(cumulative_activations, transcoder_activations):
                    cumulative += current
            
        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # null the gradients
        model.zero_grad(set_to_none=True)
            
        # calculate *approximate* cumulative time and step average for logging during training
        # Note: This is approximate because it includes the time for the current step's forward/backward pass
        # The more precise time is recorded just before validation
        if master_process:
            torch.cuda.synchronize() # Ensure accurate timing up to this point
            # Calculate time elapsed since the end of the last validation phase
            current_segment_duration_ms = 1000 * (time.perf_counter() - t0) 
            # Calculate the *true* approximate cumulative time
            approx_cumulative_time_ms = training_time_ms + current_segment_duration_ms
            approx_step_avg_ms = approx_cumulative_time_ms / (step + 1)
            print0(master_process, logfile, f"step:{step+1}/{args.train_steps} "
                    f"train_loss:{loss.item():4e} "
                    f"train_time:{approx_cumulative_time_ms:.0f}ms "
                    f"step_avg:{approx_step_avg_ms:.2f}ms", console=True)
            
            # Log training step timing to CSV
            if metrics_csv_path:
                with open(metrics_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Loss is not typically calculated per training step here, add loss logging if needed
                    writer.writerow([step + 1, "train", "", f"{approx_cumulative_time_ms:.0f}", f"{approx_step_avg_ms:.2f}"])


    print0(master_process, logfile, f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

    if world_size > 1:
        dist.destroy_process_group()
            
if __name__ == "__main__":
    main()