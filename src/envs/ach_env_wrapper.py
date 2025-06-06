#!/usr/bin/env python3
"""
ACH論文準拠の二人麻雀環境ラッパー
観測空間: 4×16の画像形式（各牌の枚数を表現）
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, List, Optional
import sys
import os

# 既存のコードをインポート
try:
    from .mahjong_game import MahjongGame, GameMode, GameResult
    from .mahjong_tile import Tile, TileType, Hand
except ImportError:
    from mahjong_game import MahjongGame, GameMode, GameResult
    from mahjong_tile import Tile, TileType, Hand


class TwoPlayerMahjongEnvACH(gym.Env):
    """
    ACH論文仕様の二人麻雀環境
    観測空間: 4×4×16の形式
    - 4チャンネル: 自分の捨て牌、相手の捨て牌、自分の手牌、相手の手牌
    - 4×16画像: 各牌の枚数を4×16の画像として表現
    - 16種類の牌: 9種類の数牌（1-9萬）+ 7種類の字牌（東南西北白發中）
    """
    
    def __init__(self, simplified: bool = True, oracle_mode: bool = False):
        super().__init__()
        
        # 簡素化モードを使用（萬子と字牌のみ）
        self.game_mode = GameMode.SIMPLIFIED if simplified else GameMode.NORMAL
        self.oracle_mode = oracle_mode  # Oracleモード（相手の手牌も見える）
        self.game = None
        
        # 観測空間の定義（ACH論文準拠: 4×4×16）
        # チャンネル: [自分の捨て牌, 相手の捨て牌, 自分の手牌, 相手の手牌]
        # 各チャンネルは4×16の画像（牌の枚数を画像形式で表現）
        self.observation_shape = (4, 4, 16)  # 4チャンネル × 4×16画像
        self.observation_space = spaces.Box(
            low=0, high=1,  # 画像形式の二値データ
            shape=self.observation_shape, 
            dtype=np.float32
        )
        
        # 行動空間の定義（打牌のみに簡略化）
        # 0-15: 各牌を捨てる（16種類）
        self.action_space = spaces.Discrete(16)
        
        # 牌のマッピング（簡素化モード用）
        # 0-8: 1-9萬
        # 9-15: 東南西北白發中
        self.tile_mapping = {}
        self.reverse_mapping = {}
        self._create_tile_mapping()
        
        self.reset()
    
    def _create_tile_mapping(self):
        """牌とインデックスのマッピングを作成"""
        # 数牌（萬子のみ）
        for i in range(9):
            tile = Tile(TileType.MAN, i + 1)
            self.tile_mapping[str(tile)] = i
            self.reverse_mapping[i] = tile
        
        # 字牌
        honors = ["東", "南", "西", "北", "白", "發", "中"]
        for i, honor in enumerate(honors):
            tile = Tile(TileType.HONOR, honor=honor)
            self.tile_mapping[str(tile)] = 9 + i
            self.reverse_mapping[9 + i] = tile
    
    def reset(self) -> np.ndarray:
        """環境をリセット"""
        self.game = MahjongGame(self.game_mode)
        self.game.start_game()
        
        # 報酬を初期化
        self.episode_rewards = [0.0, 0.0]
        self.done = False
        
        # 初期観測を返す
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        行動を実行（打牌のみ）
        action: 0-15の整数（捨てる牌の種類）
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}
        
        current_player_idx = self.game.current_player
        current_player = self.game.get_current_player()
        reward = 0.0
        info = {
            "current_player": current_player_idx,
            "action": action,
            "valid": False
        }
        
        # 行動を実行（指定された種類の牌を捨てる）
        if 0 <= action < 16:
            target_tile = self.reverse_mapping.get(action)
            if target_tile and target_tile in current_player.tiles:
                success = self.game.discard_tile(target_tile)
                info["valid"] = success
                
                if not success:
                    reward = -0.01  # 無効な行動へのペナルティ
            else:
                # 持っていない牌を捨てようとした
                reward = -0.01
                info["valid"] = False
                # 代わりに最初の牌を捨てる
                if current_player.tiles:
                    self.game.discard_tile(current_player.tiles[0])
        
        # ゲーム終了チェック
        # ツモ和了チェック（自動）
        if self.game.check_win(self.game.get_current_player()):
            self.done = True
            reward = 1.0
            info["win_type"] = "tsumo"
        elif self.game.is_game_over():
            self.done = True
            final_rewards = self.game.end_game()
            
            # 最終報酬を取得
            for i in range(2):
                self.episode_rewards[i] = final_rewards[f"player_{i}"]
            
            # 現在のプレイヤーの最終報酬を返す
            reward = self.episode_rewards[current_player_idx]
        
        # 次の観測を取得
        obs = self._get_observation()
        
        return obs, reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        現在の状態から観測を生成（4×4×16の画像形式）
        観測形状: (4, 4, 16)
        - チャンネル0: 自分の捨て牌の4×16画像
        - チャンネル1: 相手の捨て牌の4×16画像
        - チャンネル2: 自分の手牌の4×16画像
        - チャンネル3: 相手の手牌の4×16画像（通常は0、Oracleモード時は実際の値）
        
        各画像の行:
        - 1行目: その牌を1枚持っているかどうか
        - 2行目: その牌を2枚持っているかどうか
        - 3行目: その牌を3枚持っているかどうか
        - 4行目: その牌を4枚持っているかどうか
        """
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        
        if self.game is None:
            return obs
        
        current_player_idx = self.game.current_player
        opponent_idx = 1 - current_player_idx
        
        # チャンネル0: 自分の捨て牌
        self._fill_discard_tiles(obs, 0, self.game.players[current_player_idx].discarded)
        
        # チャンネル1: 相手の捨て牌
        self._fill_discard_tiles(obs, 1, self.game.players[opponent_idx].discarded)
        
        # チャンネル2: 自分の手牌
        self._fill_hand_tiles(obs, 2, self.game.players[current_player_idx].tiles)
        
        # チャンネル3: 相手の手牌（Oracleモード時のみ実際の値、通常は0）
        if self.oracle_mode:
            self._fill_hand_tiles(obs, 3, self.game.players[opponent_idx].tiles)
        # else: 既にゼロで初期化されているのでそのまま
        
        return obs
    
    def _fill_hand_tiles(self, obs: np.ndarray, channel: int, tiles: List) -> None:
        """
        手牌の情報を観測配列に設定
        """
        # 牌の枚数をカウント
        tile_counts = {}
        for tile in tiles:
            tile_str = str(tile)
            tile_counts[tile_str] = tile_counts.get(tile_str, 0) + 1
        
        # 各牌について、枚数に応じて対応する行に1を設定
        for tile_str, count in tile_counts.items():
            idx = self.tile_mapping.get(tile_str)
            if idx is not None:
                # 1枚持っている場合は1行目、2枚なら2行目、3枚なら3行目、4枚なら4行目
                for i in range(1, min(count + 1, 5)):  # 1-4枚の範囲
                    obs[channel, i - 1, idx] = 1.0
    
    def _fill_discard_tiles(self, obs: np.ndarray, channel: int, discarded_tiles: List) -> None:
        """
        捨て牌の情報を観測配列に設定
        """
        # 捨て牌の枚数をカウント
        tile_counts = {}
        for tile in discarded_tiles:
            tile_str = str(tile)
            tile_counts[tile_str] = tile_counts.get(tile_str, 0) + 1
        
        # 各牌について、枚数に応じて対応する行に1を設定
        for tile_str, count in tile_counts.items():
            idx = self.tile_mapping.get(tile_str)
            if idx is not None:
                # 1枚捨てている場合は1行目、2枚なら2行目、3枚なら3行目、4枚なら4行目
                for i in range(1, min(count + 1, 5)):  # 1-4枚の範囲
                    obs[channel, i - 1, idx] = 1.0
    
    def get_valid_actions(self) -> np.ndarray:
        """
        現在の状態で有効な行動を取得
        Returns: 有効な行動を示すマスク（1: 有効, 0: 無効）
        """
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        
        if self.game is None or self.done:
            return mask
        
        current_player = self.game.get_current_player()
        
        # 手牌にある牌の種類をマーク
        for tile in current_player.tiles:
            idx = self.tile_mapping.get(str(tile))
            if idx is not None:
                mask[idx] = 1.0
        
        # 有効な行動が一つもない場合（エラー回避）
        if np.sum(mask) == 0:
            mask[0] = 1.0  # デフォルトで最初の行動を有効にする
        
        return mask
    
    def render(self, mode='human', show_actual_state=False):
        """現在の状態を表示
        
        Args:
            mode: 表示モード
            show_actual_state: True の場合、実際のゲーム状態も表示
        """
        if self.game is None:
            return
        
        obs = self._get_observation()
        
        print("\n" + "="*50)
        print(f"現在のプレイヤー: {self.game.current_player + 1}")
        print(f"山牌残り: {self.game.deck.remaining_count()}枚")
        print(f"Oracle モード: {self.oracle_mode}")
        
        if show_actual_state:
            self._render_actual_state()
        else:
            self._render_observation_state(obs)
        
        # ゲーム状態の詳細
        state = self.game.get_game_state()
        print(f"\nリーチ状態: {state['riichi_declared']}")
        print(f"現在の報酬: {[f'{r:.2f}' for r in state['rewards']]}")
    
    def _render_observation_state(self, obs):
        """観測データに基づく状態表示"""
        print("\n--- 観測データ（プレイヤーから見える情報） ---")
        
        tile_names = ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                      '東', '南', '西', '北', '白', '發', '中']
        
        channels = ['自分の捨て牌', '相手の捨て牌', '自分の手牌', '相手の手牌']
        
        for ch_idx, channel_name in enumerate(channels):
            print(f"\n{channel_name}:")
            tiles_info = []
            for tile_idx in range(16):
                # 4×16画像の各列の合計を計算（その牌の総枚数）
                count = int(np.sum(obs[ch_idx, :, tile_idx]))
                if count > 0:
                    tiles_info.append(f"{tile_names[tile_idx]}×{count}")
            
            if tiles_info:
                print("  " + " ".join(tiles_info))
            else:
                if ch_idx == 3 and not self.oracle_mode:
                    print("  非表示（Oracle モードではない）")
                else:
                    print("  なし")
    
    def _render_actual_state(self):
        """実際のゲーム状態を表示"""
        print("\n--- 実際のゲーム状態（デバッグ用） ---")
        
        current_player_idx = self.game.current_player
        opponent_idx = 1 - current_player_idx
        
        tile_names = ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                      '東', '南', '西', '北', '白', '發', '中']
        
        # プレイヤー0の情報
        print(f"\nプレイヤー1（{current_player_idx == 0 and '自分' or '相手'}）:")
        self._print_player_tiles(self.game.players[0], tile_names)
        
        # プレイヤー1の情報
        print(f"\nプレイヤー2（{current_player_idx == 1 and '自分' or '相手'}）:")
        self._print_player_tiles(self.game.players[1], tile_names)
    
    def _print_player_tiles(self, player, tile_names):
        """プレイヤーの牌情報を表示"""
        # 手牌
        hand_counts = {}
        for tile in player.tiles:
            tile_str = str(tile)
            hand_counts[tile_str] = hand_counts.get(tile_str, 0) + 1
        
        hand_info = []
        for tile_str, count in hand_counts.items():
            hand_info.append(f"{tile_str}×{count}")
        
        print(f"  手牌（{len(player.tiles)}枚）: {' '.join(hand_info) if hand_info else 'なし'}")
        
        # 捨て牌
        discard_counts = {}
        for tile in player.discarded:
            tile_str = str(tile)
            discard_counts[tile_str] = discard_counts.get(tile_str, 0) + 1
        
        discard_info = []
        for tile_str, count in discard_counts.items():
            discard_info.append(f"{tile_str}×{count}")
        
        print(f"  捨て牌（{len(player.discarded)}枚）: {' '.join(discard_info) if discard_info else 'なし'}")
    
    def get_observation_details(self) -> Dict:
        """観測の詳細情報を取得（デバッグ用）"""
        obs = self._get_observation()
        details = {
            "observation_shape": obs.shape,
            "channel_names": ["自分の捨て牌", "相手の捨て牌", "自分の手牌", "相手の手牌"],
            "tile_names": ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                          '東', '南', '西', '北', '白', '發', '中'],
            "counts": {},
            "oracle_mode": self.oracle_mode,
            "actual_hand_sizes": {
                "player_0": len(self.game.players[0].tiles),
                "player_1": len(self.game.players[1].tiles)
            }
        }
        
        for ch_idx, ch_name in enumerate(details["channel_names"]):
            details["counts"][ch_name] = {}
            for tile_idx in range(16):
                # 4×16画像の各列の合計を計算（その牌の総枚数）
                count = int(np.sum(obs[ch_idx, :, tile_idx]))
                if count > 0:
                    tile_name = details["tile_names"][tile_idx]
                    details["counts"][ch_name][tile_name] = count
        
        return details
    
    def get_actual_game_state(self) -> Dict:
        """実際のゲーム状態を取得（デバッグ用）"""
        if self.game is None:
            return {}
        
        current_player_idx = self.game.current_player
        opponent_idx = 1 - current_player_idx
        
        state = {
            "current_player": current_player_idx,
            "oracle_mode": self.oracle_mode,
            "deck_remaining": self.game.deck.remaining_count(),
            "player_0": {
                "hand_size": len(self.game.players[0].tiles),
                "hand_tiles": [str(tile) for tile in self.game.players[0].tiles],
                "discarded_size": len(self.game.players[0].discarded),
                "discarded_tiles": [str(tile) for tile in self.game.players[0].discarded]
            },
            "player_1": {
                "hand_size": len(self.game.players[1].tiles),
                "hand_tiles": [str(tile) for tile in self.game.players[1].tiles],
                "discarded_size": len(self.game.players[1].discarded),
                "discarded_tiles": [str(tile) for tile in self.game.players[1].discarded]
            }
        }
        
        return state


# テストコード
if __name__ == "__main__":
    print("=== ACH論文準拠の二人麻雀環境テスト ===")
    
    # 環境の初期化
    env = TwoPlayerMahjongEnvACH(simplified=True)
    
    print(f"観測空間: {env.observation_space}")
    print(f"観測の形状: {env.observation_space.shape}")
    print(f"行動空間: {env.action_space}")
    print(f"行動数: {env.action_space.n}")
    
    # エピソード実行
    obs = env.reset()
    print(f"\n初期観測:\n{obs}")
    
    # 詳細情報の表示
    details = env.get_observation_details()
    print("\n観測の詳細:")
    for ch_name, counts in details["counts"].items():
        if counts:
            print(f"{ch_name}: {counts}")
    
    # 環境の表示（観測データ）
    env.render()
    
    # 実際のゲーム状態の表示
    env.render(show_actual_state=True)
    
    # 数ステップ実行
    print("\n=== ゲームプレイ ===")
    done = False
    step_count = 0
    total_reward = 0
    
    while not done and step_count < 50:
        # 有効な行動を取得
        valid_actions = env.get_valid_actions()
        valid_indices = np.where(valid_actions > 0)[0]
        
        if len(valid_indices) > 0:
            # ランダムに有効な行動を選択
            action = np.random.choice(valid_indices)
        else:
            action = 0
        
        # ステップ実行
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if info.get("valid", False):
            tile_name = details["tile_names"][action]
            print(f"ステップ {step_count + 1}: {tile_name}を捨てた, 報酬={reward:.2f}")
            
            if step_count % 10 == 0:
                env.render(show_actual_state=True)
        
        step_count += 1
    
    print(f"\nエピソード終了: 総ステップ数={step_count}, 総報酬={total_reward:.2f}")
    
    # 最終状態を表示
    env.render(show_actual_state=True)