# train_diffusion.py  ── full-feature diffusion trainer (v5)

from __future__ import annotations
import argparse, json, os, time, math, csv, random
import torch, torch.nn.functional as F
import torch.distributed as dist

# --- GPT-Lab helpers --------------------------------------------------
from train_gpt import (
    Muon,
    distributed_data_generator,
    print0,
    CastedLinear,
    Rotary,
    norm,
)

# --- Diffusion modules ------------------------------------------------
from diffusion.model import DiffusionGPT, DiffusionConfig
from diffusion.scheduler import MaskScheduler, MaskScheduleCfg

# ---------------------------------------------------------------------#
# Argument parser (mirrors original train_gpt.py) ---------------------#
# ---------------------------------------------------------------------#
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_files", type=str, required=True)
    p.add_argument("--val_files", type=str, required=True)
    p.add_argument("--train_seq_len", type=int, default=8192)
    p.add_argument("--val_seq_len", type=int,   default=8192)
    p.add_argument("--train_steps", type=int,   default=10_000)
    p.add_argument("--val_steps",   type=int,   default=100)
    p.add_argument("--grad_acc_steps", type=int, default=1)

    # diffusion options
    p.add_argument("--schedule", default="uniform",
                   choices=["uniform", "linear", "cosine", "deterministic"])
    p.add_argument("--remask",   default="random",
                   choices=["random", "lowconf", "top_p", "hybrid", "none"])
    p.add_argument("--rb_loss", action="store_true")
    p.add_argument("--infer_scaling", action="store_true")
    p.add_argument("--top_p",  type=float, default=0.9)
    p.add_argument("--det_seed", type=int,  default=0)
    p.add_argument("--target_accept", type=float, default=0.9)

    # optimisation
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--mu_lr",  type=float, default=0.02)
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--run_tag", default="")        # <<< so --run_tag is legal
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--model_name", default="diffusion_checkpoint.pt")
    return p.parse_args()


# ---------------------------------------------------------------------#
# Utility: build optimiser split --------------------------------------#
# ---------------------------------------------------------------------#
def make_optim(model: torch.nn.Module, lr: float, mu_lr: float):
    mu_params, other = [], []
    for n, p in model.named_parameters():
        (mu_params if p.ndim == 2 else other).append(p)
    return [
        Muon(mu_params, lr=mu_lr),
        torch.optim.AdamW(other, lr=lr, fused=True),
    ]


# ---------------------------------------------------------------------#
# Main training routine ----------------------------------------------#
# ---------------------------------------------------------------------#
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    cuda_id = 0
    device = torch.device(f'cuda:{cuda_id}')
    #device = torch.device('cpu')

    # ---- DDP rank / world -------------------------------------------
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    master_process = rank == 0

    # ---- logfile setup ----------------------------------------------
    log_path   = None
    csv_writer = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"diffusion_log_rank{rank}.csv")
        # -------------- keep a live CSV writer ---------------------------
        log_handle = open(log_path, "w", newline="")
        csv_writer = csv.writer(log_handle)
        csv_writer.writerow(["step", "train_loss", "val_loss",
                            "lr", "tokens/s", "elapsed_s"])
        log_handle.flush()

    # ---- model & scheduler ------------------------------------------
    cfg = DiffusionConfig()
    model = DiffusionGPT(cfg)
    opt_list = make_optim(model, args.lr, args.mu_lr)
    model = torch.compile(model).to(device)

    sched_cfg = MaskScheduleCfg(
        num_steps=cfg.num_steps,
        mask_token_id=cfg.mask_token_id,
        schedule=args.schedule,
        remask=args.remask,
        top_p=args.top_p,
        deterministic_seed=args.det_seed,
        use_inference_scaling=args.infer_scaling,
        target_accept=args.target_accept,
    )
    scheduler = MaskScheduler(sched_cfg)

    # ---- data loaders -----------------------------------------------
    train_iter = distributed_data_generator(
        master_process, log_path, args, args.train_files, args.train_seq_len, rank, world
    )
    val_iter = distributed_data_generator(
        master_process, log_path, args, args.val_files, args.val_seq_len, rank, world, print_stats=False
    )

    # ---- training loop ----------------------------------------------
    tokens_processed = 0
    best_val = float("inf")
    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        model.train()
        batch, _ = next(train_iter)
        batch = batch.to(device)
        tokens_processed += batch.numel() * world

        B = batch.size(0)
        t = torch.randint(0, cfg.num_steps, (B,), device=device)
        xt, mask = scheduler.q_sample(batch, t)

        loss = model(
            xt,
            #t,
            targets=batch,
            rb_loss=args.rb_loss,
            mask=mask,
        ) / args.grad_acc_steps
        loss.backward()

        if step % args.grad_acc_steps == 0:
            for o in opt_list:
                o.step()
                o.zero_grad(set_to_none=True)

        # ---- logging every 100 steps --------------------------------
        if step % 100 == 0 and master_process:
            elapsed = time.time() - t0
            tok_per_s = tokens_processed / max(elapsed, 1e-6)
            lr = opt_list[-1].param_groups[0]["lr"]
            print0(
                master_process,
                log_path,
                f"step {step}/{args.train_steps} | loss {loss.item():.4f} "
                f"| lr {lr:.2e} | tok/s {tok_per_s:.0f}",
                console=True,
            )

        # ---- validation ---------------------------------------------
        if step % args.val_steps == 0:
            model.eval()
            with torch.no_grad():
                vloss = 0.0
                for _ in range(10):  # small val loop
                    vb, _ = next(val_iter)
                    vb = vb.to()
                    Bv = vb.size(0)
                    tv = torch.randint(0, cfg.num_steps, (Bv,), device=device)
                    vx, msk = scheduler.q_sample(vb, tv)
                    vloss += model(vx, tv, targets=vb, rb_loss=args.rb_loss, mask=msk).item()
                vloss /= 10

            if master_process:
                elapsed = time.time() - t0
                lr = opt_list[-1].param_groups[0]["lr"]
                tok_per_s = tokens_processed / max(elapsed, 1e-6)
                csv.writer(log_handle).writerow([step, loss.item(), vloss, lr, tok_per_s, elapsed])
                log_handle.flush()

                print0(
                    master_process,
                    log_path,
                    f"[VAL] step {step} | val_loss {vloss:.4f}",
                    console=True,
                )

                # best-checkpoint
                if args.save_model and vloss < best_val:
                    best_val = vloss
                    torch.save(model.state_dict(), args.model_name)
                    print0(master_process, log_path, "saved best model", console=True)

    if log_handle:
        log_handle.close()


# ---------------------------------------------------------------------#
if __name__ == "__main__":
    main()
