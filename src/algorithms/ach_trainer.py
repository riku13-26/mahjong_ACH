import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
import os

# 経験タプル（論文準拠）
Experience = namedtuple('Experience', [
    'state',           # 状態 s
    'action',          # 行動 a
    'reward',          # 報酬 r
    'next_state',      # 次状態 s'
    'done',            # 終了フラグ
    'old_log_prob',    # データ収集時の対数確率 log π_old(a|s)
    'value',           # データ収集時の価値 V(s)
    'returns',         # 累積報酬 G
    'advantages',      # Advantage A(s,a)
    'old_regret',      # データ収集時の累積後悔推定値 y(a|s; θ_old)
    'valid_mask'       # 合法手マスク
])

class TrajectoryBuffer:
    """軌跡ベースのバッファ（ACH用に拡張）"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.returns = []
        self.advantages = []
        self.old_regrets = []  # 累積後悔推定値を追加
        self.valid_masks = []  # 合法手マスク
    
    def add(self, state, action, reward, value, log_prob, done, old_regret, valid_mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.old_regrets.append(old_regret)
        self.valid_masks.append(valid_mask)
    
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """GAE(λ)を使用したAdvantageの計算"""
        self.returns = [0] * len(self.rewards)
        self.advantages = [0] * len(self.rewards)
        
        # 逆順で計算
        running_return = 0
        running_advantage = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            # TD誤差
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # GAE
            running_advantage = delta + gamma * gae_lambda * (1 - self.dones[t]) * running_advantage
            self.advantages[t] = running_advantage
            
            # Returns
            running_return = self.rewards[t] + gamma * (1 - self.dones[t]) * running_return
            self.returns[t] = running_return
    
    def get_batch(self):
        """バッチデータを返す"""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'returns': np.array(self.returns),
            'advantages': np.array(self.advantages),
            'values': np.array(self.values),
            'old_regrets': np.array(self.old_regrets),
            'valid_masks': np.array(self.valid_masks)
        }
    
    def clear(self):
        """バッファをクリア"""
        self.__init__()

class ACHTrainer:
    """
    論文準拠のACH (Actor-Critic Hedge) トレーナー
    NW-CFRの実践的実装
    """
    def __init__(
        self,
        env,
        network_class,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        eta=1.0,  # Hedge温度パラメータ
        logit_threshold=2.0,  # l_th
        max_grad_norm=0.5,
        update_epochs=4,
        batch_size=64,
        device=None
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.eta = eta
        self.logit_threshold = logit_threshold
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ネットワーク
        self.network = network_class(
            obs_shape=env.observation_space.shape,
            n_actions=env.action_space.n,
            logit_threshold=logit_threshold
        ).to(self.device)
        
        # オプティマイザ
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 統計
        self.episode_rewards = []
        self.training_stats = []
    
    def select_action(self, state, deterministic=False):
        """
        Hedgeによる行動選択
        π(a|s) = exp(η * y(a|s; θ)) / Σ exp(η * y(a|s; θ))
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 有効な行動マスク (batch 次元を持たせ、デバイスへ転送)
            valid_actions = self.env.get_valid_actions()             # list/np.array 0–1
            valid_mask = torch.FloatTensor(valid_actions).unsqueeze(0).to(self.device)

            # ネットワーク推論（合法手マスクを渡す）
            regret_estimates, policy_logits, value = self.network(
                state_tensor,
                valid_mask=valid_mask,
                eta=self.eta
            )

            # 有効行動以外を除外した logits
            masked_logits = policy_logits.clone()
            masked_logits = masked_logits.masked_fill(valid_mask == 0, -1e9)

            # Hedgeによる方策
            probs = F.softmax(masked_logits, dim=-1)
            
            if deterministic:
                action = probs.argmax().item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
            
            log_prob = torch.log(probs[0, action])
            
            # 選択された行動の累積後悔推定値
            old_regret = regret_estimates[0, action].item()
            
            # CPU 上の 0/1 list を返す
            valid_actions_list = valid_actions  # already obtained before
            
            return action, log_prob.item(), value.item(), old_regret, valid_actions_list
    
    def collect_trajectories(self, n_steps=2048):
        """軌跡の収集（論文のActorに相当）"""
        buffer = TrajectoryBuffer()
        state = self.env.reset()
        episode_reward = 0
        
        for _ in range(n_steps):
            # 行動選択
            action, log_prob, value, old_regret, valid_mask = self.select_action(state)
            
            # 環境でステップ実行
            next_state, reward, done, _ = self.env.step(action)
            
            # バッファに追加
            buffer.add(state, action, reward, value, log_prob, done, old_regret, valid_mask)
            
            episode_reward += reward
            
            if done:
                self.episode_rewards.append(episode_reward)
                state = self.env.reset()
                episode_reward = 0
            else:
                state = next_state
        
        # Advantage計算
        buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
        
        return buffer
    
    def compute_ach_loss(self, batch):
        """ACH 損失関数を計算して総損失と統計を返す"""

        # --- 1) バッチデータを GPU / CPU のテンソルへ変換 ----------------------
        states        = torch.FloatTensor(batch['states']).to(self.device)       # 状態 s_t
        actions       = torch.LongTensor(batch['actions']).to(self.device)       # 行動 a_t
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)    # 収集時 log π_old(a|s)
        returns       = torch.FloatTensor(batch['returns']).to(self.device)      # 割引累積報酬 G_t
        advantages    = torch.FloatTensor(batch['advantages']).to(self.device)   # GAE で推定した A(s,a)
        old_regrets   = torch.FloatTensor(batch['old_regrets']).to(self.device)  # （保存用）旧 y(a|s)
        valid_actions_mask = torch.FloatTensor(batch['valid_masks']).to(self.device)  # 合法手マスク

        # --- 2) ネットワーク推論 -------------------------------------------------
        regret_estimates, policy_logits, values = self.network(
            states,
            valid_mask=valid_actions_mask,
            eta=self.eta
        )

        # --- 4) 不合法行動の logit を −∞ にしてソフトマックスから除外 -------------
        masked_logits = torch.where(valid_actions_mask==1, policy_logits, torch.full_like(policy_logits, -1e9))

        # --- 5) 現在ポリシーの log π(a|s; θ) と重要度比 ---------------------------
        log_probs         = F.log_softmax(masked_logits, dim=-1)                  # 全行動の log π
        current_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # 選択行動のみ
        ratio             = torch.exp(current_log_probs - old_log_probs)          # π / π_old
        ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)

        # --- 6) 選択行動の累積後悔 y_a と π_old(a|s) ----------------------------
        y_a      = regret_estimates.gather(1, actions.unsqueeze(1)).squeeze(1)    # y(a|s; θ)
        pi_old_a = old_log_probs.exp()                                            # π_old(a|s)

        # --- 7) c 係数の判定 (両方のクリップ条件を満たすと 1) ----------------------
        clip_hi, clip_lo  = 1.0 + self.clip_epsilon, 1.0 - self.clip_epsilon      # PPO 型閾値
        within_ratio      = torch.where(advantages >= 0,                          # A>=0 なら上側制限
                                         ratio < clip_hi,                         #  A<0 なら下側制限
                                         ratio > clip_lo)
        within_yclip      = y_a.abs() < self.logit_threshold                      # |y_a| <= l_th
        c                 = (within_ratio & within_yclip).float()                 # bool → {0,1}

        # --- 8) Advantage を標準化して学習を安定化 --------------------------------
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 9) Actor 損失 (Eq.29) ----------------------------------------------
        actor_term = -(self.eta * c * (y_a / (pi_old_a + 1e-8)) * advantages)      # サンプル毎
        actor_loss = actor_term.mean()                                             # バッチ平均

        # ---10) Critic 損失 (MSE) -----------------------------------------------
        value_loss = F.mse_loss(values, returns)

        # ---11) ポリシー・エントロピー (探索促進) ---------------------------------
        probs = torch.exp(log_probs) * valid_actions_mask  # zero out invalid actions
        safe_log = torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))
        entropy = -(probs * safe_log).sum(dim=-1).mean()

        # ---12) 総損失の合成 ------------------------------------------------------
        total_loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # ---13) 学習モニタリング用の統計値 ---------------------------------------
        stats = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy':    entropy.item(),
            'mean_ratio': ratio.mean().item(),
            'mean_regret': y_a.mean().item()
        }

        # ---14) 呼び出し元へ返却 --------------------------------------------------
        return total_loss, stats

    
    def update_network(self, buffer):
        """ネットワークの更新（論文のLearnerに相当）"""
        batch_data = buffer.get_batch()
        n_samples = len(batch_data['states'])
        
        # 複数エポックでの更新
        for epoch in range(self.update_epochs):
            # ミニバッチでの更新
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                # ミニバッチの作成
                mini_batch = {
                    key: value[batch_indices] for key, value in batch_data.items()
                }
                
                # 損失計算
                loss, stats = self.compute_ach_loss(mini_batch)
                
                # 勾配更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.training_stats.append(stats)
        
        # 統計情報を返す
        if self.training_stats:
            latest_stats = self.training_stats[-1].copy()
            latest_stats['avg_reward'] = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            return latest_stats
        else:
            return {
                'actor_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'mean_ratio': 1.0,
                'avg_reward': 0.0
            }
    
    def train(self, total_timesteps=1000000, log_interval=10):
        """メイン訓練ループ"""
        timesteps = 0
        iteration = 0
        
        while timesteps < total_timesteps:
            # 軌跡収集
            buffer = self.collect_trajectories(n_steps=2048)
            timesteps += 2048
            
            # ネットワーク更新
            self.update_network(buffer)
            
            # ログ出力
            if iteration % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_stats = {
                    key: np.mean([s[key] for s in self.training_stats[-100:]])
                    for key in self.training_stats[0].keys()
                } if self.training_stats else {}
                
                print(f"\nIteration {iteration}, Timesteps {timesteps}")
                print(f"  Average Reward: {avg_reward:.3f}")
                if avg_stats:
                    print(f"  Actor Loss: {avg_stats['actor_loss']:.4f}")
                    print(f"  Value Loss: {avg_stats['value_loss']:.4f}")
                    print(f"  Entropy: {avg_stats['entropy']:.4f}")
                    print(f"  Mean Ratio: {avg_stats['mean_ratio']:.4f}")
                    print(f"  Mean Regret: {avg_stats['mean_regret']:.4f}")
            
            buffer.clear()
            iteration += 1
        
        print("Training completed!")
    
    def evaluate(self, n_episodes=100):
        """評価"""
        rewards = []
        wins = 0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _, _, _, _ = self.select_action(state, deterministic=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            if episode_reward > 0:
                wins += 1
        
        print(f"\nEvaluation Results ({n_episodes} episodes):")
        print(f"  Average Reward: {np.mean(rewards):.3f}")
        print(f"  Win Rate: {wins/n_episodes:.1%}")
        
        return np.mean(rewards)
    
    def save_checkpoint(self, filepath):
        """チェックポイント保存"""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'training_stats': self.training_stats[-1000:],
            'eta': self.eta
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """チェックポイント読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.training_stats = checkpoint.get('training_stats', [])
        self.eta = checkpoint.get('eta', self.eta)
        print(f"Checkpoint loaded: {filepath}")
