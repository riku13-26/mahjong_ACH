#!/usr/bin/env python3
"""
簡易麻雀ゲームの対話型テストプログラム
プレイヤーが手動で牌を選択してプレイできます
"""

from mahjong_game import MahjongGame, GameMode, GameResult
from mahjong_tile import Tile, TileType
import sys


def display_hand(hand, player_name):
    """手牌を見やすく表示"""
    print(f"\n{player_name}の手牌 ({hand.count_tiles()}枚):")
    tiles_str = " ".join([f"[{i+1}]{str(tile)}" for i, tile in enumerate(hand.tiles)])
    print(tiles_str)


def get_player_choice(hand, player_name, game, player_index):
    """プレイヤーの選択を取得"""
    while True:
        try:
            display_hand(hand, player_name)
            
            # リーチ済みの場合は自動打牌
            if hand.riichi_declared:
                print(f"{player_name}はリーチ済みです。自動で最初の牌を捨てます。")
                return 0
            
            # テンパイ時のリーチ選択
            is_tenpai = game.is_tenpai(hand)
            if is_tenpai and not hand.riichi_declared:
                print(f"\n{player_name}はテンパイです！")
                waiting_tiles = game.get_waiting_tiles(hand)
                print(f"待ち牌: {', '.join(str(tile) for tile in waiting_tiles)}")
                
                # フリテンチェック
                is_furiten = game.is_furiten(player_index)
                if is_furiten:
                    print("⚠️ フリテン状態です（ロン和了不可）")
                
                riichi_choice = input("リーチしますか？ (y/n): ").strip().lower()
                if riichi_choice == 'y':
                    if game.declare_riichi(player_index):
                        print(f"{player_name}がリーチを宣言しました！（-0.1点）")
                        # リーチ後は自動打牌
                        return 0
                    else:
                        print("リーチできませんでした。")
            
            choice = input(f"{player_name}、捨てる牌を選んでください (1-{len(hand.tiles)}): ").strip()
            
            tile_index = int(choice) - 1
            if 0 <= tile_index < len(hand.tiles):
                return tile_index
            else:
                print("無効な選択です。")
        except ValueError:
            print("数字を入力してください。")


def check_ron_opportunity(game, discarded_tile, discarding_player_idx):
    """ロン和了の機会をチェック"""
    other_player_idx = 1 - discarding_player_idx
    other_player = game.players[other_player_idx]
    
    if game.can_ron(other_player_idx, discarded_tile):
        player_name = f"プレイヤー{other_player_idx + 1}"
        print(f"\n🎉 {player_name}がロン和了できます！")
        
        ron_choice = input(f"{player_name}、ロンしますか？ (y/n): ").strip().lower()
        if ron_choice == 'y':
            if game.ron_win(other_player_idx, discarded_tile):
                print(f"🎉 {player_name}のロン和了！")
                results = game.end_game()
                print("最終報酬:")
                for player, reward in results.items():
                    print(f"  {player}: {reward:.1f}")
                return True
    return False

def display_game_state(game):
    """ゲーム状態を表示"""
    state = game.get_game_state()
    print("\n" + "="*50)
    print(f"ターン: プレイヤー{state['current_player'] + 1}")
    print(f"山牌残り: {state['deck_remaining']}枚")
    print(f"ドラ表示牌: {', '.join(state['dora_indicators'])}")
    print(f"現在の報酬: {[f'{r:.1f}' for r in state['rewards']]}")
    print(f"リーチ状況: {['宣言済み' if r else '未宣言' for r in state['riichi_declared']]}")
    
    # 各プレイヤーの捨て牌表示
    for i, player_data in enumerate(state['player_hands']):
        if player_data['discarded']:
            print(f"プレイヤー{i+1}の捨て牌: {' '.join(player_data['discarded'])}")  # 最新5枚のみ表示


def check_tenpai_status(game):
    """各プレイヤーのテンパイ状況をチェック"""
    print("\n--- テンパイ状況 ---")
    for i, player in enumerate(game.players):
        is_tenpai = game.is_tenpai(player)
        status = "テンパイ" if is_tenpai else "ノーテン"
        print(f"プレイヤー{i+1}: {status}")


def main():
    print("=== 簡易麻雀ゲーム（対話型） ===")
    print("簡素化ルール:")
    print("- 萬子と字牌（三元牌・風牌）のみ")
    print("- 鳴きなし")
    print("- リーチあり（テンパイ時に選択可能）")
    print("\n操作方法:")
    print("- 数字（1-14）: 捨てる牌を選択")
    print("- テンパイ時: y/nでリーチ選択")
    print("- 相手の捨て牌でロン和了可能")
    
    # ゲーム初期化
    game = MahjongGame(GameMode.SIMPLIFIED)
    game.start_game()
    
    print(f"\nゲーム開始！")
    display_game_state(game)
    
    turn_count = 0
    max_turns = 100  # 無限ループ防止
    
    while not game.is_game_over() and turn_count < max_turns:
        current_player_idx = game.current_player
        current_player = game.get_current_player()
        player_name = f"プレイヤー{current_player_idx + 1}"
        
        # 和了チェック
        if game.check_win(current_player):
            print(f"\n🎉 {player_name}の和了！")
            results = game.end_game()
            print("最終報酬:")
            for player, reward in results.items():
                print(f"  {player}: {reward:.1f}")
            break
        
        # プレイヤーの選択
        tile_index = get_player_choice(current_player, player_name, game, current_player_idx)
        
        # 牌を捨てる
        discard_tile = current_player.tiles[tile_index]
        success = game.discard_tile(discard_tile)
        
        if success:
            print(f"{player_name}が {discard_tile} を捨てました")
            
            # ロン和了チェック
            if check_ron_opportunity(game, discard_tile, current_player_idx):
                break
            
            turn_count += 1
            display_game_state(game)
        else:
            print("牌を捨てることができませんでした。")
    
    # ゲーム終了処理
    if turn_count >= max_turns:
        print(f"\n{max_turns}ターン経過により強制終了")
    elif game.deck.remaining_count() == 0:
        print("\n流局（山牌が尽きました）")
        check_tenpai_status(game)
        results = game.end_game()
        print("最終報酬:")
        for player, reward in results.items():
            print(f"  {player}: {reward:.1f}")
    
    # 最終状態表示
    final_state = game.get_game_state()
    print(f"\nゲーム終了 - 総ターン数: {turn_count}")
    print(f"最終報酬: {[f'{r:.1f}' for r in final_state['rewards']]}")
    print("プレイヤーの手牌")
    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nゲームが中断されました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)