# src/algorithms/train.py
"""
ACH論文ベースの麻雀AI訓練メインスクリプト
リーチ無しの簡素化ルールで学習
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import argparse
import random
from datetime import datetime

from envs.ach_env_wrapper import TwoPlayerMahjongEnvACH
from algorithms.ach_trainer import ACHTrainer

def set_seed(seed: int = 42) -> None:
    """再現性のためのシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")
              
def create_environment(oracle_mode: bool = False, simplified: bool = True):
    """環境の作成ヘルパー"""
    env = TwoPlayerMahjongEnvACH(
        simplified=simplified,
        oracle_mode=oracle_mode,
    )

    print("Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space:      {env.action_space.n}")
    print(f"  Oracle mode:       {oracle_mode}")
    print(f"  Simplified mode:   {simplified}")
    return env

def main():
    parser = argparse.ArgumentParser(description='Train ACH Mahjong AI')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eta', type=float, default=0.1, help='Hedge temperature parameter')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--oracle_mode', action='store_true', help='Use oracle mode (see opponent hand)')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Checkpoint file to load')
    parser.add_argument('--collect_freq', type=int, default=100, help='Experience collection frequency')
    parser.add_argument('--eval_freq', type=int, default=1000, help='Evaluation frequency')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = create_environment(oracle_mode=args.oracle_mode)
    
    # Device setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create ACH trainer
    trainer = ACHTrainer(
        env=env,
        obs_shape=(4, 4, 16),
        n_action=16,  # リーチ無し、16の打牌行動のみ
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        eta=args.eta,  # Hedge温度パラメータ
        device=device
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Training or evaluation
    if args.eval_only:
        print("Running evaluation only...")
        trainer.evaluate(num_episodes=100, render=True)
    else:
        print("Starting training...")
        start_time = datetime.now()
        
        try:
            trainer.train(
                num_iterations=args.iterations,
                collect_freq=args.collect_freq,
                eval_freq=args.eval_freq
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"Training completed in {training_time}")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        trainer.evaluate(num_episodes=100, render=False)
        
        # Save final model
        final_checkpoint = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        trainer.save_checkpoint(final_checkpoint)
        print(f"Final model saved as {final_checkpoint}")

if __name__ == "__main__":
    main()