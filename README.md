# 二人麻雀AI環境 (Two-Player Mahjong AI Environment)

OpenAI Gym準拠の二人麻雀環境実装。強化学習による麻雀AIの研究・開発を目的としています。

## 特徴

- **OpenAI Gym準拠**: 標準的な強化学習フレームワークと互換性
- **画像形式観測**: CNNベースのモデルに最適化された(4, 4, 16)形状の観測空間
- **Oracleモード**: デバッグ・研究用に相手の手牌も観測可能
- **簡素化モード**: 萬子と字牌のみを使用した学習しやすい環境
- **完全な麻雀ルール**: 和了判定、リーチ、基本的な麻雀ルールを実装

## インストール

### 必要な環境
- Python 3.7+
- gym
- numpy
- matplotlib (可視化用)

### セットアップ

```bash
git clone https://github.com/your-username/mahjong_ai_ACH.git
cd mahjong_ai_ACH
pip install -r requirements.txt
```

## 基本的な使用方法

### 1. 環境の初期化

```python
from src.envs.ach_env_wrapper import TwoPlayerMahjongEnvACH

# 通常モード（相手の手牌は観測不可）
env = TwoPlayerMahjongEnvACH(simplified=True, oracle_mode=False)

# Oracleモード（相手の手牌も観測可能、デバッグ用）
env_oracle = TwoPlayerMahjongEnvACH(simplified=True, oracle_mode=True)
```

### 2. 基本的なゲームループ

```python
import numpy as np

# 環境をリセット
obs = env.reset()
print(f"観測形状: {obs.shape}")  # (4, 4, 16)

done = False
total_reward = 0

while not done:
    # 有効な行動を取得
    valid_actions_mask = env.get_valid_actions()
    valid_actions = np.where(valid_actions_mask > 0)[0]
    
    # ランダムに行動を選択
    if len(valid_actions) > 0:
        action = np.random.choice(valid_actions)
    else:
        action = 0
    
    # 行動を実行
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    print(f"行動: {action}, 報酬: {reward}, 終了: {done}")

print(f"総報酬: {total_reward}")
```

### 3. 状態の可視化

```python
# 観測データベースの表示
env.render()

# 実際のゲーム状態表示（デバッグ用）
env.render(show_actual_state=True)
```

## 観測空間

環境は`(4, 4, 16)`形状の観測を返します：

- **4チャンネル**: 自分の捨て牌、相手の捨て牌、自分の手牌、相手の手牌
- **4×16画像**: 各牌の枚数を4×16の画像として表現
- **16種類の牌**: 9種類の数牌（1-9萬）+ 7種類の字牌（東南西北白發中）

### チャンネル構成

| チャンネル | 内容 | 通常モード | Oracleモード |
|------------|------|------------|--------------|
| 0 | 自分の捨て牌 | ✓ | ✓ |
| 1 | 相手の捨て牌 | ✓ | ✓ |
| 2 | 自分の手牌 | ✓ | ✓ |
| 3 | 相手の手牌 | ✗ (0) | ✓ |

### 画像形式の表現

各チャンネルの4×16画像：
- **行0**: その牌を1枚持っている場合に1
- **行1**: その牌を2枚持っている場合に1
- **行2**: その牌を3枚持っている場合に1  
- **行3**: その牌を4枚持っている場合に1

## 行動空間

16種類の離散行動（各牌を捨てる行動）：
- **0-8**: 1-9萬を捨てる
- **9-15**: 東南西北白發中を捨てる

## サンプルコード

### CNNモデルでの学習例

```python
import torch
import torch.nn as nn

class MahjongCNN(nn.Module):
    def __init__(self, num_actions=16):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 16, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 使用例
env = TwoPlayerMahjongEnvACH(simplified=True)
model = MahjongCNN()

obs = env.reset()
obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
q_values = model(obs_tensor)

# 有効な行動のみを考慮
valid_actions = env.get_valid_actions()
q_values_masked = q_values.clone()
q_values_masked[0, valid_actions == 0] = -float('inf')
action = q_values_masked.argmax().item()
```

### 詳細な状態分析

```python
# 観測の詳細情報を取得
details = env.get_observation_details()
print("観測の詳細:")
for channel_name, counts in details["counts"].items():
    if counts:
        print(f"{channel_name}: {counts}")

# 実際のゲーム状態を取得
actual_state = env.get_actual_game_state()
print(f"プレイヤー0の手牌枚数: {actual_state['player_0']['hand_size']}")
print(f"プレイヤー1の手牌枚数: {actual_state['player_1']['hand_size']}")
```

## API リファレンス

### `TwoPlayerMahjongEnvACH`

#### コンストラクタ
```python
TwoPlayerMahjongEnvACH(simplified=True, oracle_mode=False)
```

**パラメータ:**
- `simplified` (bool): 簡素化モード（萬子と字牌のみ）
- `oracle_mode` (bool): Oracleモード（相手の手牌も観測可能）

#### 主要メソッド

##### `reset() -> np.ndarray`
環境をリセットし、初期観測を返す。

##### `step(action: int) -> Tuple[np.ndarray, float, bool, Dict]`
行動を実行し、次の観測、報酬、終了フラグ、追加情報を返す。

##### `get_valid_actions() -> np.ndarray`
現在の状態で有効な行動のマスクを返す。

##### `render(mode='human', show_actual_state=False)`
現在の状態を表示する。
- `show_actual_state=True`: 実際のゲーム状態（デバッグ用）も表示

##### `get_observation_details() -> Dict`
観測の詳細情報を返す。

##### `get_actual_game_state() -> Dict`
実際のゲーム状態を返す（デバッグ用）。

## ファイル構成

```
mahjong_ai_ACH/
├── src/
│   ├── envs/
│   │   ├── ach_env_wrapper.py      # メイン環境クラス
│   │   ├── mahjong_game.py         # 麻雀ゲームロジック
│   │   ├── mahjong_tile.py         # 牌とデッキの実装
│   │   └── test_ach_env.ipynb      # テスト用ノートブック
│   ├── algorithms/                 # 学習アルゴリズム
│   └── utils/                      # ユーティリティ
├── requirements.txt                # 依存関係
├── README.md                       # このファイル
└── tests/                          # テストファイル
```

## テスト

```bash
# 基本テスト
python src/envs/ach_env_wrapper.py

# Jupyter Notebookでのインタラクティブテスト
jupyter notebook src/envs/test_ach_env.ipynb
```

## 技術的詳細

### 麻雀ルール
- 二人麻雀ルール
- 簡素化モード：萬子（1-9萬）と字牌（東南西北白發中）のみ
- 基本的な和了判定（一般形、七対子）
- リーチ機能

### 強化学習向け設計
- 連続的な状態表現（画像形式）
- スパースでない報酬設計
- 有効行動マスクによる無効行動の除外
- エピソード終了条件の明確化

## 今後の拡張予定

- [ ] フル麻雀ルール（筒子・索子の追加）
- [ ] 鳴き（ポン・チー・カン）の実装
- [ ] より複雑な役・点数計算
- [ ] 四人麻雀対応
- [ ] 事前訓練済みモデルの提供

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告を歓迎します。大きな変更を加える場合は、まずissueで議論してください。

## 引用

この環境を研究で使用する場合は、以下のように引用してください：

```bibtex
@misc{mahjong_ai_ach,
  title={Two-Player Mahjong AI Environment},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-username/mahjong_ai_ACH}
}
```

## 関連研究

- [ACH: Agent-based Monte Carlo Tree Search for Mahjong](https://example.com) (参考論文)
- [OpenAI Gym](https://gym.openai.com/)
- [強化学習による麻雀AI](https://example.com)