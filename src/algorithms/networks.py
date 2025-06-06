import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """ResNet-style residual block"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)

class ACHNetwork(nn.Module):
    """
    ACH論文 Figure 7 構成:
        - 画像型特徴 (4×4×16) → 3-stage Residual CNN 64→128→32ch
        - フラット化 + (必要に応じ one‑hot特徴を後段で連結)
        - FC(1024) 共有融合層
        - Actor / Critic それぞれ FC(512)×2 → 出力
    """
    def __init__(
        self,
        obs_shape=(4, 4, 16),
        n_actions=16,
        logit_threshold=2.0,
        onehot_dim: int = 0  # 追加の one‑hot ベクトル長 (0 のままなら無し)
    ):
        super().__init__()

        if len(obs_shape) != 3:
            raise ValueError("obs_shape must be (H, W, C)")

        h, w, c = obs_shape
        self.h, self.w, self.n_actions = h, w, n_actions
        self.logit_threshold = logit_threshold
        self.onehot_dim = onehot_dim

        # === Stage‑1 : C=64 ===
        self.conv_in = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(*[ResidualBlock(64) for _ in range(3)])

        # === Transition 1 → 128 ch ===
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(*[ResidualBlock(128) for _ in range(3)])

        # === Transition 2 → 32 ch ===
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(*[ResidualBlock(32) for _ in range(3)])

        # === Shared FC fusion ===
        conv_flat_dim = 32 * h * w
        fusion_in_dim = conv_flat_dim + onehot_dim
        self.fusion_fc = nn.Linear(fusion_in_dim, 1024)

        # === Actor head ===
        self.actor_fc1 = nn.Linear(1024, 512)
        self.actor_fc2 = nn.Linear(512, 512)
        self.actor_out = nn.Linear(512, n_actions)

        # === Critic head ===
        self.critic_fc1 = nn.Linear(1024, 512)
        self.critic_fc2 = nn.Linear(512, 512)
        self.value_out = nn.Linear(512, 1)

        self._initialize_weights()

    # -------------------------------------------------------------

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # 初期累積後悔 0 付近
        nn.init.xavier_uniform_(self.actor_out.weight, gain=0.1)
        nn.init.constant_(self.actor_out.bias, 0)

    # -------------------------------------------------------------

    def forward(self, img_obs, valid_mask=None, onehot_feat=None, eta: float = 1.0):
        """
        Args:
            img_obs: (B, 4, 4, 16) 画像特徴
            valid_mask: (B, n_actions) BoolTensor – 中央化平均で合法手のみ計算
            onehot_feat: (B, onehot_dim) 追加ベクトル（無ければ None）
            eta: Hedge 温度
        Returns:
            regret_estimates, policy_logits, value
        """
        # (B, H, W, C) -> (B, C, H, W)
        x = img_obs.permute(0, 3, 1, 2).contiguous()

        # Backbone
        x = self.conv_in(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)

        # Flatten conv output
        flat = x.reshape(x.size(0), -1)

        # Concatenate optional one‑hot features
        if self.onehot_dim:
            if onehot_feat is None:
                raise ValueError("onehot_feat must be provided when onehot_dim>0")
            flat = torch.cat([flat, onehot_feat], dim=-1)

        # Shared fusion layer
        fused = F.relu(self.fusion_fc(flat))

        # === Actor branch ===
        a = F.relu(self.actor_fc1(fused))
        a = F.relu(self.actor_fc2(a))
        regret_estimates = self.actor_out(a)

        # ---- Centralisation w.r.t valid_mask ----
        if valid_mask is not None:
            valid_mask_bool = valid_mask.bool()
            masked = regret_estimates.masked_fill(~valid_mask_bool, 0.0)
            cnt = valid_mask_bool.sum(dim=1, keepdim=True).clamp(min=1)
            mean = masked.sum(dim=1, keepdim=True) / cnt
        else:
            mean = regret_estimates.mean(dim=1, keepdim=True)
        regret_estimates = regret_estimates - mean

        # Clip
        regret_estimates = torch.clamp(regret_estimates,
                                       -self.logit_threshold,
                                       self.logit_threshold)
        # Hedge logits
        policy_logits = eta * regret_estimates

        # === Critic branch ===
        c = F.relu(self.critic_fc1(fused))
        c = F.relu(self.critic_fc2(c))
        value = self.value_out(c).squeeze(-1)

        return regret_estimates, policy_logits, value