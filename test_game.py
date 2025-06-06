#!/usr/bin/env python3
from mahjong_game import MahjongGame, GameMode, GameResult
from mahjong_tile import Tile, TileType


def test_basic_game():
    print("=== 二人麻雀ゲームテスト ===")
    
    # ゲーム初期化（通常モード）
    game = MahjongGame(GameMode.NORMAL)
    print("ゲームを開始します...")
    game.start_game()
    
    # ゲーム状態表示
    state = game.get_game_state()
    print(f"現在のプレイヤー: {state['current_player']}")
    print(f"山牌残り: {state['deck_remaining']}枚")
    print(f"ドラ表示牌: {state['dora_indicators']}")
    
    # 各プレイヤーの手牌表示
    for i, player_data in enumerate(state['player_hands']):
        print(f"\nプレイヤー{i+1}の手牌 ({player_data['tile_count']}枚):")
        print(" ".join(player_data['tiles']))
    
    # ゲームプレイのシミュレーション
    turn_count = 0
    while not game.is_game_over() and turn_count < 20:
        current_player = game.get_current_player()
        print(f"\n--- ターン {turn_count + 1} (プレイヤー{game.current_player + 1}) ---")
        
        # 手牌から適当な牌を捨てる
        if current_player.tiles:
            discard_tile = current_player.tiles[0]
            print(f"プレイヤー{game.current_player + 1}が {discard_tile} を捨てました")
            game.discard_tile(discard_tile)
            print(f"現在の手牌: {state['player_hands'][game.current_player]['tiles']}")
        
        # 和了チェック
        for i, player in enumerate(game.players):
            if game.check_win(player):
                print(f"\nプレイヤー{i+1}の和了！")
                return
        
        turn_count += 1
    
    print(f"\nゲーム終了 ({turn_count}ターン経過)")
    
    # 最終状態表示
    final_state = game.get_game_state()
    print(f"山牌残り: {final_state['deck_remaining']}枚")


def test_tile_creation():
    print("\n=== 牌作成テスト ===")
    
    # 各種牌のテスト
    tiles = [
        Tile(TileType.MAN, 1),
        Tile(TileType.PIN, 5),
        Tile(TileType.SOU, 9),
        Tile(TileType.HONOR, honor="東"),
        Tile(TileType.HONOR, honor="白")
    ]
    
    print("作成された牌:")
    for tile in tiles:
        print(f"  {tile}")


def test_winning_hand():
    print("\n=== 和了判定テスト ===")
    
    game = MahjongGame()
    
    # テスト用の和了手を作成（七対子）
    test_tiles = [
        Tile(TileType.MAN, 1), Tile(TileType.MAN, 1),
        Tile(TileType.MAN, 2), Tile(TileType.MAN, 2),
        Tile(TileType.MAN, 3), Tile(TileType.MAN, 3),
        Tile(TileType.PIN, 1), Tile(TileType.PIN, 1),
        Tile(TileType.PIN, 2), Tile(TileType.PIN, 2),
        Tile(TileType.SOU, 1), Tile(TileType.SOU, 1),
        Tile(TileType.HONOR, honor="東"), Tile(TileType.HONOR, honor="東")
    ]
    
    print("テスト手牌（七対子）:")
    print(" ".join(str(tile) for tile in test_tiles))
    
    is_win = game._is_winning_hand(test_tiles)
    print(f"和了判定: {'和了' if is_win else '未和了'}")


def test_simplified_game():
    print("\n=== 簡素化モードテスト ===")
    
    # 簡素化モードでゲーム初期化
    game = MahjongGame(GameMode.SIMPLIFIED)
    game.start_game()
    
    state = game.get_game_state()
    print(f"モード: {state['mode']}")
    print(f"鳴き有効: {state['player_hands'][0]['allow_melds']}")
    print(f"山牌数: {state['deck_remaining']}枚 (簡素化モードでは減少)")
    
    # 簡素化モードでは萬子・字牌のみが配られることを確認
    all_tiles = []
    for player_data in state['player_hands']:
        all_tiles.extend(player_data['tiles'])
    
    has_pin = any('筒' in tile for tile in all_tiles)
    has_sou = any('索' in tile for tile in all_tiles)
    print(f"筒子が含まれている: {has_pin} (簡素化モードではFalseのはず)")
    print(f"索子が含まれている: {has_sou} (簡素化モードではFalseのはず)")


def test_reward_system():
    print("\n=== 報酬システムテスト ===")
    
    game = MahjongGame(GameMode.SIMPLIFIED)
    
    # 各種報酬のテスト
    rewards = {
        "win": game.calculate_reward(0, GameResult.WIN),
        "lose": game.calculate_reward(0, GameResult.LOSE),
        "tenpai": game.calculate_reward(0, GameResult.DRAW_TENPAI),
        "noten": game.calculate_reward(0, GameResult.DRAW_NOTEN)
    }
    
    print(f"和了報酬: {rewards['win']}")
    print(f"放銃報酬: {rewards['lose']}")
    print(f"テンパイ報酬: {rewards['tenpai']}")
    print(f"ノーテン報酬: {rewards['noten']}")
    
    # リーチテスト
    print("\nリーチテスト:")
    print(f"初期報酬: {game.rewards}")
    
    # リーチ宣言をシミュレート（実際にはテンパイチェックが必要）
    game.riichi_declared[0] = True
    game.rewards[0] -= 0.1
    print(f"リーチ後報酬: {game.rewards}")
    
    # リーチ者の和了時報酬
    riichi_win_reward = game.calculate_reward(0, GameResult.WIN)
    print(f"リーチ者の和了報酬: {riichi_win_reward} (+0.2のボーナス)")


def test_mode_comparison():
    print("\n=== モード比較テスト ===")
    
    # 通常モード
    normal_game = MahjongGame(GameMode.NORMAL)
    normal_game.start_game()
    normal_state = normal_game.get_game_state()
    
    # 簡素化モード
    simplified_game = MahjongGame(GameMode.SIMPLIFIED)
    simplified_game.start_game()
    simplified_state = simplified_game.get_game_state()
    
    print(f"通常モード - 山牌数: {normal_state['deck_remaining']}枚")
    print(f"簡素化モード - 山牌数: {simplified_state['deck_remaining']}枚")
    
    print(f"通常モード - 鳴き有効: {normal_state['player_hands'][0]['allow_melds']}")
    print(f"簡素化モード - 鳴き有効: {simplified_state['player_hands'][0]['allow_melds']}")


if __name__ == "__main__":
    test_tile_creation()
    test_winning_hand()
    test_basic_game()
    test_simplified_game()
    test_reward_system()
    test_mode_comparison()