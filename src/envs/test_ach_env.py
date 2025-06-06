#!/usr/bin/env python3
"""
ACH環境の観測を可視化するツール
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from ach_env_wrapper import TwoPlayerMahjongEnvACH


def visualize_observation(obs, title="ACH環境の観測"):
    """
    4×16の観測データを可視化
    
    Args:
        obs: 観測データ (4, 16)のnumpy配列
        title: 図のタイトル
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    
    # 牌の名前
    tile_names = ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                  '東', '南', '西', '北', '白', '發', '中']
    
    # チャンネル名
    channel_names = ['自分の手牌', '自分の捨て牌', '相手の捨て牌', '相手の手牌（推定）']
    
    # 各チャンネルを可視化
    for ch_idx in range(4):
        ax = axes[ch_idx]
        
        # 棒グラフで枚数を表示
        x = np.arange(16)
        bars = ax.bar(x, obs[ch_idx], color='steelblue', edgecolor='black')
        
        # 枚数が0より大きい場合、数値を表示
        for i, (bar, count) in enumerate(zip(bars, obs[ch_idx])):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(count)}', ha='center', va='bottom')
        
        # 軸の設定
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(0, 5)
        ax.set_xticks(x)
        ax.set_xticklabels(tile_names, rotation=45, ha='right')
        ax.set_ylabel('枚数')
        ax.set_title(channel_names[ch_idx])
        ax.grid(True, axis='y', alpha=0.3)
        
        # 数牌と字牌の境界線
        ax.axvline(x=8.5, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def visualize_observation_heatmap(obs, title="ACH環境の観測（ヒートマップ）"):
    """
    4×16の観測データをヒートマップで可視化
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # ヒートマップを作成
    im = ax.imshow(obs, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # 軸の設定
    tile_names = ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                  '東', '南', '西', '北', '白', '發', '中']
    channel_names = ['自分の手牌', '自分の捨て牌', '相手の捨て牌', '相手の手牌']
    
    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(tile_names)
    ax.set_yticklabels(channel_names)
    
    # 各セルに数値を表示
    for i in range(4):
        for j in range(16):
            count = int(obs[i, j])
            if count > 0:
                text = ax.text(j, i, count, ha="center", va="center", 
                             color="white" if count > 2 else "black")
    
    # 数牌と字牌の境界線
    ax.axvline(x=8.5, color='blue', linewidth=2)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('枚数', rotation=270, labelpad=20)
    
    ax.set_title(title)
    plt.tight_layout()
    return fig


def compare_observations(obs1, obs2, title1="観測1", title2="観測2"):
    """
    2つの観測を比較表示
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    tile_names = ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                  '東', '南', '西', '北', '白', '發', '中']
    channel_names = ['自分の手牌', '自分の捨て牌', '相手の捨て牌', '相手の手牌']
    
    # 観測1を上段に表示
    for ch_idx in range(4):
        ax = axes[0, ch_idx]
        x = np.arange(16)
        bars = ax.bar(x, obs1[ch_idx], color='steelblue', alpha=0.7)
        
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_title(channel_names[ch_idx])
        if ch_idx == 0:
            ax.set_ylabel(title1)
    
    # 観測2を下段に表示
    for ch_idx in range(4):
        ax = axes[1, ch_idx]
        x = np.arange(16)
        bars = ax.bar(x, obs2[ch_idx], color='coral', alpha=0.7)
        
        ax.set_ylim(0, 5)
        ax.set_xticks(x)
        ax.set_xticklabels(tile_names, rotation=45, ha='right')
        if ch_idx == 0:
            ax.set_ylabel(title2)
    
    plt.suptitle('観測の比較', fontsize=16)
    plt.tight_layout()
    return fig


def analyze_game_progress(env, num_steps=20):
    """
    ゲームの進行を分析し、観測の変化を可視化
    """
    # ゲームをリセット
    obs = env.reset()
    observations = [obs.copy()]
    actions = []
    rewards = []
    
    # 指定ステップ数だけゲームを進める
    for step in range(num_steps):
        # 有効な行動を取得
        valid_actions = env.get_valid_actions()
        valid_indices = np.where(valid_actions > 0)[0]
        
        if len(valid_indices) > 0:
            action = np.random.choice(valid_indices)
        else:
            action = 0
        
        # ステップ実行
        obs, reward, done, info = env.step(action)
        
        if info['valid']:
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
        
        if done:
            break
    
    # 観測の変化を可視化
    fig, axes = plt.subplots(len(observations), 1, figsize=(14, 4 * len(observations)))
    
    tile_names = ['1萬', '2萬', '3萬', '4萬', '5萬', '6萬', '7萬', '8萬', '9萬',
                  '東', '南', '西', '北', '白', '發', '中']
    
    for i, obs in enumerate(observations):
        ax = axes[i] if len(observations) > 1 else axes
        
        # 4チャンネルを結合して表示
        combined = np.zeros((4, 16))
        combined[0] = obs[0]  # 自分の手牌（正の値）
        combined[1] = -obs[1]  # 自分の捨て牌（負の値）
        combined[2] = -obs[2] * 0.5  # 相手の捨て牌（薄い負の値）
        
        # カスタムカラーマップで表示
        im = ax.imshow(combined[:3], cmap='RdBu', aspect='auto', 
                      vmin=-4, vmax=4, interpolation='nearest')
        
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['手牌', '自捨牌', '相手捨牌'])
        ax.set_xticks(np.arange(16))
        ax.set_xticklabels(tile_names, rotation=45, ha='right')
        
        if i == 0:
            ax.set_title('初期状態')
        else:
            action_tile = tile_names[actions[i-1]]
            ax.set_title(f'ステップ {i}: {action_tile}を捨てた (報酬: {rewards[i-1]:.2f})')
    
    plt.suptitle('ゲーム進行の観測変化', fontsize=16)
    plt.tight_layout()
    return fig


# テストコード
if __name__ == "__main__":
    print("=== ACH環境の観測可視化テスト ===")
    
    # 環境を作成
    env = TwoPlayerMahjongEnvACH(simplified=True)
    
    # 初期観測を取得
    obs = env.reset()
    
    # 基本的な可視化
    print("1. 棒グラフによる可視化")
    fig1 = visualize_observation(obs, "初期状態の観測")
    plt.show()
    
    # ヒートマップによる可視化
    print("\n2. ヒートマップによる可視化")
    fig2 = visualize_observation_heatmap(obs, "初期状態の観測（ヒートマップ）")
    plt.show()
    
    # ゲームを少し進めて比較
    print("\n3. ゲーム進行による観測の変化")
    for _ in range(5):
        valid_actions = env.get_valid_actions()
        valid_indices = np.where(valid_actions > 0)[0]
        if len(valid_indices) > 0:
            action = np.random.choice(valid_indices)
            env.step(action)
    
    obs_after = env._get_observation()
    fig3 = compare_observations(obs, obs_after, "初期状態", "5手後")
    plt.show()
    
    # ゲーム進行の分析
    print("\n4. ゲーム進行の詳細分析")
    env.reset()  # リセット
    fig4 = analyze_game_progress(env, num_steps=10)
    plt.show()
    
    print("\n可視化テスト完了")