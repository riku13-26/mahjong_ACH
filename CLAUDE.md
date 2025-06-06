# Mahjong AI Project Memory

## Project Overview
This is a mahjong AI implementation using ACH (Actor-Critic Hedge) algorithm for learning optimal play strategies. The project implements a simplified two-player mahjong environment with neural network-based agents.

## Project Structure
### Core Components
- `src/envs/mahjong_game.py`: Core mahjong game logic and rules
- `src/envs/mahjong_tile.py`: Tile representation and utilities  
- `src/envs/ach_env_wrapper.py`: ACH-compliant environment wrapper (4×4×16 observation space)
- `src/algorithms/ach_trainer.py`: ACH trainer implementation with Hedge-based action selection
- `src/algorithms/networks.py`: Neural network architectures
- `src/algorithms/train.py`: Training utilities and configurations

### Training and Results
- `ach_train.py`: Main training script
- `results/checkpoints/`: Saved model checkpoints
- `results/figures/`: Training curves and plots
- `notebooks/test_ach_trainer.ipynb`: Training experiments notebook

### Tests
- `test_game.py`: Basic game logic tests
- `test_simplified_game.py`: Simplified game mode tests
- `test_win_check.py`: Win condition validation tests
- `src/envs/test_ach_env.py`: Environment wrapper tests

## Environment Specifications
### Observation Space
- Shape: (4, 4, 16) - 4 channels × 4×16 image format
- Channel 0: Own discarded tiles
- Channel 1: Opponent's discarded tiles  
- Channel 2: Own hand tiles
- Channel 3: Opponent's hand tiles (hidden in normal mode, visible in Oracle mode)
- 16 tile types: 9 Man tiles (1-9萬) + 7 Honor tiles (東南西北白發中)

### Action Space
- Discrete(16): Discard one of 16 possible tile types
- Valid action masking based on current hand

### Reward Structure
- Win: +1.0
- Loss: -1.0
- Draw (Tenpai): +0.2
- Draw (Noten): 0.0
- Invalid action: -0.01
- Riichi declaration: -0.1 (returned +0.2 on win)

## ACH Algorithm Details
- Uses Hedge-based policy: π(a|s) = exp(η * y(a|s; θ)) / Σ exp(η * y(a|s; θ))
- Regret estimation with logit threshold clipping
- PPO-style ratio clipping for stability
- GAE (Generalized Advantage Estimation) for advantage calculation

## Coding Standards
- Use Python 3.x
- Follow PEP 8 style guidelines
- Include comprehensive docstrings for functions and classes
- Write unit tests for new functionality

## Training
- Run training with: `python ach_train.py`
- Monitor with Weights & Biases logging
- Checkpoints saved automatically during training

## Testing
- Run tests with: `python test_game.py`
- Environment tests: `python src/envs/test_ach_env.py`
- Ensure all tests pass before committing changes

## Dependencies
- Install dependencies with: `pip install -r requirements.txt`
- Key dependencies: torch, gym, numpy, wandb
- Use virtual environment located in `venv/`