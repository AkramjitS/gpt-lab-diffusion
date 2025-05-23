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
import tiktoken
import json
import datetime
import pickle
import shutil
import csv
import random
import math
import numpy as np # Import numpy for potential future use, set random seed now not to forget to set it later
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
#torch._inductor.config.max_autotune_gemm_backends = ["ATEN"]

@dataclass
class Hyperparameters:
    """
    default values are set to fit on a 2x GPUs w/ 8GB of VRAM each, but are not necessarily optimal
    """
    model_name = "ModdedGPT"
    # data
    train_files = "data/fineweb*_train_*.bin" # input .bin to train on
    val_files = "data/fineweb*_val_*.bin" # input .bin to eval validation loss on
    train_seq_len = 8*1024 # FlexAttention sequence length
    val_seq_len = 16*1024 # FlexAttention sequence length for validation (should be able to fit more than train_seq_len)
    # optimization loop
    val_steps = 10 # number of steps to run validation for
    train_steps = 20#_000 # number of training steps to run
    grad_acc_steps = 1 # number of gradient accumulation steps per training step
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    tokenizer = "gpt4regex_v50256_n1000000000.pkl"# any .pkl file in tokenizers/
    vocab_size = 50257 # should be the tokenizer's size plus any special tokens
    # model size - parameters set for GPUs w/ 8GB VRAM
    num_layers = 12  # number of reansformer blocks
    num_heads = 6   # number of attention heads
    model_dim = 384  # size of model embedding vectors
    head_dim = None  # size of attention heads; if None, will default to model_dim // num_heads
    mlp_ratio = 4  # MLP hidden dimension is model_dim * mlp_ratio
    num_val_emb = 2 # number of value embeddings used at initial and final layers
    # memory optimization 
    use_fp8 = False # experimental; True on H100s (and newer?) should improve performance but seems to use more vram somehow
    # evaluation and logging
    val_loss_every = 100 # every how many steps to evaluate val loss? 0 for only at the end
    save_model = False
    # reproducibility
    seed: int | None = None # Optional random seed for initialization control

    # my variables
    cuda: int = 1
    hellaswag_validation: bool = False
    mask_token_id: int = 50255
    num_generate_steps: int = 50

    def __post_init__(self):
        # Validate and set derived parameters
        assert self.train_seq_len % 128 == 0, f"train_seq_len must be multiple of 128, got {self.train_seq_len}"
        assert self.val_seq_len % 128 == 0, f"val_seq_len must be multiple of 128, got {self.val_seq_len}"
        assert self.grad_acc_steps >= 1, f"grad_acc steps must be int >= 1"
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
        assert self.head_dim in [2 ** i for i in range(1, 10)], f"head_dim must be a power of 2, got {self.head_dim}"
        assert self.mlp_ratio > 0, f"mlp_ratio must be positive, got {self.mlp_ratio}"
        assert self.num_layers // 2 >= self.num_val_emb, \
            f"num_layers // 2 (={self.num_layers // 2}) must be greater than or equal num_val_emb (={self.num_val_emb})"
        assert self.num_layers % 2 == 0, f"Number of layers ({self.num_layers}) must be even for skip connections"

    @classmethod
    def from_args(cls):
        """Create Hyperparameters from command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a GPT model with customizable hyperparameters")
        
        # Data arguments
        parser.add_argument('--train_files', type=str, help='Pattern for training data files')
        parser.add_argument('--val_files', type=str, help='Pattern for validation data files')
        parser.add_argument('--train_seq_len', type=int, help='Training sequence length')
        parser.add_argument('--val_seq_len', type=int, help='Validation sequence length')
        
        # Optimization arguments
        parser.add_argument('--val_steps', type=int, help='Number of steps to run validation for')
        parser.add_argument('--train_steps', type=int, help='Number of training iterations')
        parser.add_argument('--grad_acc_steps', type=int, help='Number of gradient accumulation steps per training iteration')
        parser.add_argument('--cooldown_frac', type=float, help='Fraction of training for learning rate cooldown')
        
        # Architecture arguments
        parser.add_argument('--tokenizer', type=str, help='Tokenizer file name in tokenizers/ directory')
        parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
        parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, help='Number of attention heads')
        parser.add_argument('--model_dim', type=int, help='Model embedding dimension')
        parser.add_argument('--head_dim', type=int, help='Dimension per attention head')
        parser.add_argument('--mlp_ratio', type=int, help='MLP hidden dim ratio')
        parser.add_argument('--num_val_emb', type=int, help='Number of value embeddings used at initial and final layers')
        
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
        parser.add_argument('--hellaswag_validation', 
                            type=lambda x: (str(x).lower() == 'true'), 
                            #default=False, 
                            help='Perform HellaSwag validation after pretraining')
        parser.add_argument('--mask_token_id', type=int, 
                            #default=50255,
                            help='Default integer value for mask_token_id')
        parser.add_argument('--num_generate_steps', type=int,
                            #default=50,
                            help='Maximum generatation steps')
        
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
        files_to_copy = ["requirements.txt", sys.argv[0], "download_hellaswag.py", "download_fineweb.py"]
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

        # Handle tokenizer separately as it's a .pkl file
        tokenizer_path = Path(f"data/{args.tokenizer}")
        if tokenizer_path.exists():
            try:
                with open(tokenizer_path, 'rb') as f:
                    tokenizer_config = pickle.load(f)
                # Save the config as a pretty-printed text file
                tokenizer_log_path = experiment_dir_path / f"{tokenizer_path.stem}_config.txt"
                import pprint
                tokenizer_str = pprint.pformat(tokenizer_config)
                with open(tokenizer_log_path, "w") as f:
                    f.write(f"Tokenizer Config ({args.tokenizer}):\n")
                    f.write("="*100 + "\n")
                    f.write(tokenizer_str)
                print0(master_process, logfile, f"- Saved tokenizer config to {tokenizer_log_path}")
                del tokenizer_config # Free up memory
            except Exception as e:
                print0(master_process, logfile, f"- Error processing tokenizer {tokenizer_path}: {e}")
        else:
            print0(master_process, logfile, f"- Tokenizer file not found: {tokenizer_path}")

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
    #    Construct model and optimizer     #
    ########################################

    model: nn.Module = Diffusion(vocab_size=args.vocab_size,
                        mask_token_id=args.mask_token_id,
                        num_layers=args.num_layers,
                        num_val_emb=args.num_val_emb,
                        num_heads=args.num_heads, 
                        model_dim=args.model_dim,
                        max_seq_len=max(args.train_seq_len, args.val_seq_len),
                        mlp_ratio=args.mlp_ratio,
                        num_steps=args.num_generate_steps).cuda()
    print0(master_process, logfile, f'{model.get_num_params()} parameters', console=True)
    print0(master_process, logfile, model)

    # Set FP8 option based on hyperparameters
    #model.lm_head.use_fp8 = args.use_fp8

    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)

    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)

    # For single GPU case, we need to modify how Muon is initialized
    if world_size == 1:
        # Create update buffer for single GPU
        for param in hidden_matrix_params:
            param.requires_grad_(True)
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    else:
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.train_steps # progress in training
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1

    # Add fallback mode to handle compilation errors
    from torch import _dynamo
    torch._dynamo.config.suppress_errors = True
    
    # Use a more memory-efficient compilation option
    if args.use_fp8:
        model: nn.Module = torch.compile(model, dynamic=False)
    else:
        model: nn.Module = torch.compile(model, dynamic=False, mode="reduce-overhead")

    ########################################
    #            Warmup kernels            #
    ########################################

    print0(master_process, logfile, "warming up kernels...", console=True)

    # Attempt to limit memory fragmentation
    if hasattr(torch.cuda, 'memory_stats'):
        print0(master_process, logfile, f"Initial GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_steps = 10
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
    for _ in range(warmup_steps):
        loss = torch.tensor([0.], device="cuda")
        for _ in range(args.grad_acc_steps):
            inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda", dtype=torch.int64)
            #torch.compiler.cudagraph_mark_step_begin()
                # TODO why does un-commenting this^ line throw an error here in the warmup but not down in training?
            t = torch.randint(0, model.num_steps, (1,), device='cuda')
            step_loss = model(inputs.to(torch.int32), t, targets)
            loss += step_loss / args.grad_acc_steps
        loss.backward()
        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state # TODO optionally save initial state of model jic someone wants to test different seeds

    if hasattr(torch.cuda, 'memory_stats'):
        print0(master_process, logfile, f"After warmup GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

    print0(master_process, logfile, "kernels are toasty", console=True)

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(master_process, logfile, args, args.train_files, world_size * args.train_seq_len, rank, world_size, diffusion=True)

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    for step in range(args.train_steps + 1):
        last_step = (step == args.train_steps)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            # Note: training_time_ms accumulates *only* the time spent in the training loop
            # It does not include time spent in validation or other operations outside the loop
            training_time_ms += 1000 * (time.perf_counter() - t0)
            
            model.eval()
            
            # Ensure we validate on enough tokens while keeping memory usage reasonable
            val_batch_size = world_size * args.val_seq_len
            val_tokens_used = val_batch_size * args.val_steps
            print0(master_process, logfile, f"Validating on {val_tokens_used} tokens ({args.val_steps} steps with {val_batch_size} batch size)", console=True)
            
            val_loader = distributed_data_generator(master_process, logfile, args, args.val_files, val_batch_size, rank, world_size, print_stats=False, diffusion=True)
            val_loss = 0
            # TODO modify for validation
            with torch.no_grad():
                for i in range(args.val_steps):
                    inputs, _ = next(val_loader)
                    B = inputs.size(0)
                    # Check if inputs exceed sequence length
                    if B > args.val_seq_len:
                        inputs = inputs[:args.val_seq_len]
                    t = torch.randint(0, model.num_steps, (1,), device='cuda')
                    val_loss += model(
                        inputs,
                        t, 
                        inputs
                    )
            val_loss /= args.val_steps
            del val_loader
            if world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            
            # Calculate average time per step up to this point
            step_avg_ms = training_time_ms / max(step, 1) 
            print0(master_process, logfile, f"step:{step}/{args.train_steps} val_loss:{val_loss:.4e} "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{step_avg_ms:.2f}ms", console=True)
            
            # Log validation metrics to CSV
            if master_process and metrics_csv_path:
                with open(metrics_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Use .item() to get float from tensor for val_loss
                    writer.writerow([step, 
                        "val", f"{val_loss.item():.4f}", 
                        f"{training_time_ms:.0f}", 
                        f"{step_avg_ms:.2f}"])

            if last_step: # inside validation section to avoid the if check every training iteration
                # 5. Save model checkpoint inside the experiment directory
                if master_process and args.save_model and experiment_dir_path:
                    log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                    # Ensure experiment_dir_path exists (though it should from earlier)
                    os.makedirs(experiment_dir_path, exist_ok=True)
                    save_path = experiment_dir_path / f"state_step{step:06d}.pt"
                    torch.save(log, str(save_path))
                    print0(master_process, logfile, f"Saved checkpoint to {save_path}", console=True)
                # the last step only has the validation loop, so break to avoid training
                break
            
            model.train()
            # start the clock again for the next training segment
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # --------------- TRAINING SECTION -----------------
        loss = torch.tensor([0.], device="cuda")
        for _ in range(args.grad_acc_steps):
            inputs, _ = next(train_loader)
            B = inputs.size(0)
            # Check if inputs exceed sequence length - can happen if the dataset has different sized examples
            if B > args.train_seq_len:
                inputs = inputs[:args.train_seq_len]
            t = torch.randint(0, model.num_steps, (1,), device='cuda')
            
            torch.compiler.cudagraph_mark_step_begin()
            step_loss = model(
                inputs,
                t,
                inputs
            )
            loss += step_loss / args.grad_acc_steps
        loss.backward()
            
        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            opt.step()
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

    # After training and sample generations, evaluate on HellaSwag
    hellaswag_path = "./data/hellaswag_val.jsonl" 
    # Check if the HellaSwag data file exists
    if os.path.exists(hellaswag_path) and args.hellaswag_validation:
        print0(master_process, logfile, f"Found HellaSwag dataset at {hellaswag_path}", console=True)
        evaluate_hellaswag(master_process, logfile, world_size, rank, args, model, hellaswag_path, limit=1014, diffusion=True) # 1014 is largest possible
    else:
        print0(master_process, logfile, f"HellaSwag dataset not found at {hellaswag_path}, skipping evaluation.", console=True)

    if world_size > 1:
        dist.destroy_process_group()

    ########################################
    #        FINAL OUTPUT EXAMPLES         #
    ########################################

    def sample_from_model(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
        """Generate text samples from the model given a prompt."""
        tokenizer_config = pickle.load(open(f"tokenizers/{args.tokenizer}", 'rb'))
        enc = tiktoken.Encoding(
            name=args.tokenizer[:-4], # :-4 to remove the .pkl
            pat_str=tokenizer_config['pat_str'],
            mergeable_ranks=tokenizer_config['mergeable_ranks'],
            special_tokens={
                "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
            }
        )
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        
        # Encode the prompt
        input_ids = encode(prompt)
        x = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        # add leading token 50256
        #leading_token = torch.tensor([50256], dtype=x.dtype, device=x.device)
        #x = torch.cat([leading_token, x])

        # Generate
        model.eval()
        with torch.no_grad():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode and return
        #return decode(y[1:].tolist())
        return decode(y.tolist())

    # Then at the end of training:
    if master_process: 
        print0(master_process, logfile, "-"*10 + " EXAMPLE MODEL GENERATIONS AFTER TRAINING " + "-"*10)
        prompts = [
            "Once upon a time,",
            "The meaning of life is",
            "In the year 2026,",
            "I'm a Large Language Model (LLM), which means",
            "2 + 3 = "
        ]
        for prompt in prompts:
            continuation = sample_from_model(model, prompt, max_new_tokens=16)
            print0(master_process, logfile, continuation, console=True)
            
if __name__ == "__main__":
    main()