import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import tiktoken
import pickle
from .helper import print0

########################################
#        HellaSwag Evaluation         #
########################################

def render_hellaswag_example(example, enc):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # NOTE: prepending " " because GPT-2 based tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.int32)
    mask = torch.zeros((4, max_len), dtype=torch.int32)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def iterate_hellaswag_examples(data_path, limit=1014):
    """Iterate through HellaSwag examples, with optional limit"""
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate_hellaswag(master_process, logfile, world_size, rank, args, model, data_path, limit=1014, diffusion=False):
    """Evaluate model on HellaSwag in a distributed way using modulo distribution"""
    assert limit <= 1014, f'there are only 1014 questions in the benchmark, but got limit={limit}'
    torch._dynamo.config.disable = True
    tokenizer_config = pickle.load(open(f"tokenizers/{args.tokenizer}", 'rb'))
    enc = tiktoken.Encoding(
        name=args.tokenizer[:-4], # :-4 to remove the .pkl
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=tokenizer_config['mergeable_ranks'],
        special_tokens={
            "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
        }
    )
    model.eval()
    
    # Local counters
    local_correct_norm = 0
    local_correct = 0
    local_total = 0
    
    # Process examples that belong to this GPU (based on index % world_size)
    for i, example in enumerate(iterate_hellaswag_examples(data_path, limit)):
        # Skip examples that don't belong to this GPU
        if i % world_size != 0:
            continue

        local_total += 1
        tokens, mask, label = render_hellaswag_example(example, enc)
        tokens = tokens.to(device="cuda")
        mask = mask.to(device="cuda")

        # Process each candidate one at a time
        losses = []
        normalized_losses = []
        
        for j in range(4):  # 4 candidates per example
            # Get token sequence for this candidate
            seq = tokens[j]
            seq_mask = mask[j]
            
            # Only process up to valid tokens (not padding)
            valid_len = (seq > 0).sum().item()
            if valid_len == 0:
                continue
                
            valid_seq = seq[:valid_len]
            
            # Pad sequence to multiple of 128 for FlexAttention
            if valid_len % 128 != 0:
                # Calculate padding needed
                def cdiv(m, n):
                    return (m + (n - 1)) // n
                pad_ct = cdiv(valid_len, 128) * 128 - valid_len
                # Add padding
                valid_seq = torch.cat((valid_seq, 
                                    torch.zeros(pad_ct, dtype=valid_seq.dtype, device=valid_seq.device)), 
                                    dim=0)
            
            # Get logits from our model
            if diffusion:
                t = torch.randint(0, model.num_steps, (1,), device='cuda')
                logits = model(valid_seq, t)
            else:
                logits = model(valid_seq)
            if isinstance(logits, torch.Tensor):
                logits = logits[0]  # Our model returns [B, T, V] but B=1
            
            # We only care about the original non-padded part
            logits = logits[:valid_len]
            
            # Evaluate the autoregressive loss
            shift_logits = logits[:-1, :]
            shift_tokens = seq[1:valid_len].to(torch.int64)  # Target needs to be int64
            shift_mask = seq_mask[1:valid_len]  # Shift mask to align with shifted tokens
            
            # Calculate loss for each position
            losses_per_token = F.cross_entropy(
                shift_logits, shift_tokens, reduction='none'
            )
            
            # Apply mask to focus on completion region
            masked_losses = losses_per_token * shift_mask
            
            # Calculate total and normalized loss
            total_loss = masked_losses.sum()
            completion_token_count = shift_mask.sum()
            normalized_loss = total_loss / completion_token_count if completion_token_count > 0 else float('inf')
            
            losses.append(total_loss.item())
            normalized_losses.append(normalized_loss.item())
        
        # Get predictions and update counters
        pred = torch.tensor(losses).argmin().item()
        pred_norm = torch.tensor(normalized_losses).argmin().item()
        
        local_correct += int(pred == label)
        local_correct_norm += int(pred_norm == label)
    
    # Gather results from all processes
    correct_tensor = torch.tensor([local_correct], dtype=torch.float32, device="cuda")
    correct_norm_tensor = torch.tensor([local_correct_norm], dtype=torch.float32, device="cuda")
    total_tensor = torch.tensor([local_total], dtype=torch.float32, device="cuda")   
    
    # Handle distributed reduction
    if world_size > 1:
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_norm_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate final metrics on master process
    if master_process:
        num_correct = int(correct_tensor.item())
        num_correct_norm = int(correct_norm_tensor.item())
        num_total = int(total_tensor.item())
        
        # Calculate metrics and print results
        accuracy = num_correct / num_total if num_total > 0 else 0
        accuracy_norm = num_correct_norm / num_total if num_total > 0 else 0

        # Calculate 95% confidence intervals using Wilson score interval
        # This is more robust than normal approximation, especially for small sample sizes or extreme probabilities
        z = 1.96  # 95% confidence
        
        def wilson_conf_interval(correct, total):
            """Calculate Wilson score interval for a binary proportion"""
            if total == 0:
                return (0, 0)
            
            p = correct / total
            denominator = 1 + z**2 / total
            centre_adjusted_p = (p + z**2 / (2 * total)) / denominator
            adjusted_interval = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
            
            lower = max(0, centre_adjusted_p - adjusted_interval)
            upper = min(1, centre_adjusted_p + adjusted_interval)
            
            return (lower, upper)
        
        # Get confidence intervals
        ci = wilson_conf_interval(num_correct, num_total)
        ci_norm = wilson_conf_interval(num_correct_norm, num_total)
        
        print0(master_process, logfile, f"HellaSwag evaluation complete - {num_total} examples", console=True)
        print0(master_process, logfile, f"Accuracy: {num_correct}/{num_total}={accuracy:.4f} "
                f"[95% CI: {ci[0]:.3f}-{ci[1]:.3f}]", console=True)
        print0(master_process, logfile, f"Normalized accuracy: {num_correct_norm}/{num_total}={accuracy_norm:.4f} "
                f"[95% CI: {ci_norm[0]:.3f}-{ci_norm[1]:.3f}]", console=True)