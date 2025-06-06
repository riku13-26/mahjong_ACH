"""
ach_train.py
------------
Training launcher for ACH (Actor‑Critic Hedge) Mahjong agent.

Key points
----------
* Reward scale in your env: win +1, tenpai‑draw +0.2  (≈1/25 of paper scale)
  → learning‑rate & entropy‑coef are ×25, Hedge temperature η is ÷25.
* Designed for a single 24 GB GPU.
  n_steps = 16 384 keeps VRAM usage < 8 GB even with fp32.
* Uses Weights & Biases for live metrics & checkpoint artifacts.

Run example
------------
python ach_train.py \
    --total_steps 2000000 \
    --wandb \
    --run_name gpu24g_run1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import wandb

# ──────────────────────────────────────────────────────────────────────────
# Default hyper‑parameters (paper & scale compensation)
PAPER_LR = 1e-4
PAPER_BETA = 3e-2
PAPER_ETA = 1.0
SCALE = 25.0             # reward is 25× smaller than paper

DEF_LR = PAPER_LR * SCALE          # ≈2.5e‑3
DEF_BETA = PAPER_BETA * SCALE      # ≈0.75
DEF_ETA = PAPER_ETA / SCALE        # ≈0.04

# ──────────────────────────────────────────────────────────────────────────
def make_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("ACH trainer with wandb")
parser.add_argument("--total_steps", type=int, default=2_000_000)
parser.add_argument("--n_steps", type=int, default=16_384,
                    help="trajectory length collected before each update")
parser.add_argument("--checkpoint_steps", type=int, default=200_000)
parser.add_argument("--lr_actor", type=float, default=DEF_LR)
parser.add_argument("--lr_critic", type=float, default=DEF_LR)
parser.add_argument("--entropy_beta", type=float, default=DEF_BETA)
parser.add_argument("--eta", type=float, default=DEF_ETA)
parser.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
# wandb
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--project", type=str, default="ACH-Mahjong")
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────
# Paths
ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "results" / "checkpoints"
LOG_DIR = ROOT / "results" / "logs"
make_dirs(CKPT_DIR)
make_dirs(LOG_DIR)

# ──────────────────────────────────────────────────────────────────────────
# Weights & Biases init
if args.wandb:
    wandb_run = wandb.init(
        project=args.project,
        name=args.run_name,
        dir=str(LOG_DIR),
        config=vars(args),
        save_code=True,
    )
else:
    wandb_run = None

# ──────────────────────────────────────────────────────────────────────────
# Project‑local imports (after sys.path adjusted by package layout)
from src.envs.mahjong_env import MahjongEnv            # noqa: E402
from src.algorithms.ach_trainer import ACHTrainer      # noqa: E402

env = MahjongEnv()
trainer = ACHTrainer(
    env=env,
    device=args.device,
    lr_actor=args.lr_actor,
    lr_critic=args.lr_critic,
    entropy_coef=args.entropy_beta,
    eta=args.eta,
)

# ──────────────────────────────────────────────────────────────────────────
print("=== ACH training start ===")
print(json.dumps(vars(args), indent=2))

timesteps = 0
iteration = 0
while timesteps < args.total_steps:
    buffer = trainer.collect_trajectories(n_steps=args.n_steps)
    timesteps += args.n_steps

    metrics = trainer.update_network(buffer)

    # ── logging ────────────────────────────────────
    if iteration % 2 == 0:
        log_line = (f"Iter {iteration:4d} | steps {timesteps:8d} | "
                    f"R̄ {metrics['avg_reward']:+.3f} | "
                    f"Lπ {metrics['actor_loss']:+.4f} | "
                    f"LV {metrics['value_loss']:+.4f} | "
                    f"H {metrics['entropy']:+.3f} | "
                    f"ratio {metrics['mean_ratio']:.3f}")
        print(log_line)

    if args.wandb:
        wandb.log({"timesteps": timesteps, **metrics})

    # ── checkpoint ────────────────────────────────
    if timesteps % args.checkpoint_steps == 0 or timesteps >= args.total_steps:
        ckpt_path = CKPT_DIR / f"ach_{timesteps}.pt"
        torch.save({
            "network": trainer.network.state_dict(),
            "optim_actor": trainer.optimizer_actor.state_dict(),
            "optim_critic": getattr(trainer, "optimizer_critic", None),
            "timesteps": timesteps,
            "args": vars(args),
        }, ckpt_path)
        print(f"✅ saved checkpoint: {ckpt_path}")
        if args.wandb:
            art = wandb.Artifact(f"ach-{timesteps}", type="model")
            art.add_file(str(ckpt_path))
            wandb_run.log_artifact(art)

    iteration += 1

print("=== Training complete ===")
if args.wandb:
    wandb.finish()
