#!/usr/bin/env python3
"""
和了判定とロン和了のテストプログラム
"""

from mahjong_game import MahjongGame, GameMode
from mahjong_tile import Tile, TileType, Hand


def create_test_hand(tile_strings):
    """テスト用の手牌を作成"""
    hand = Hand(allow_melds=False)
    for tile_str in tile_strings:
        if tile_str[-1] == '萬':
            number = int(tile_str[0])
            tile = Tile(TileType.MAN, number)
        else:
            # 字牌
            tile = Tile(TileType.HONOR, honor=tile_str)
        hand.add_tile(tile)
    return hand


def test_winning_hands():
    """和了手のテスト"""
    print("=== 和了判定テスト ===")
    
    game = MahjongGame(GameMode.SIMPLIFIED)
    
    test_cases = [
        {
            "name": "七対子（正常）",
            "tiles": ["1萬", "1萬", "2萬", "2萬", "3萬", "3萬", "4萬", "4萬", "5萬", "5萬", "6萬", "6萬", "東", "東"],
            "expected": True
        },
        {
            "name": "七対子（13枚）",
            "tiles": ["1萬", "1萬", "2萬", "2萬", "3萬", "3萬", "4萬", "4萬", "5萬", "5萬", "6萬", "6萬", "東"],
            "expected": False
        },
        {
            "name": "通常和了（刻子+順子）",
            "tiles": ["1萬", "1萬", "1萬", "2萬", "3萬", "4萬", "5萬", "5萬", "6萬", "7萬", "8萬", "東", "東", "東"],
            "expected": True
        },
        {
            "name": "字牌のみ和了",
            "tiles": ["東", "東", "東", "南", "南", "南", "西", "西", "西", "北", "北", "北", "白", "白"],
            "expected": True
        },
        {
            "name": "不完全手（バラバラ）",
            "tiles": ["1萬", "3萬", "5萬", "7萬", "9萬", "東", "南", "西", "北", "白", "發", "中", "2萬", "4萬"],
            "expected": False
        },
        {
            "name": "国士無双風（13種類）",
            "tiles": ["1萬", "9萬", "東", "南", "西", "北", "白", "發", "中", "2萬", "3萬", "4萬", "5萬", "6萬"],
            "expected": False  # 通常の和了形ではない
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"手牌: {' '.join(case['tiles'])}")
        
        # 手牌作成
        hand = create_test_hand(case['tiles'])
        
        # 和了判定
        is_win = game.check_win(hand)
        
        print(f"判定結果: {'和了' if is_win else '未和了'}")
        print(f"期待結果: {'和了' if case['expected'] else '未和了'}")
        print(f"テスト: {'✅ PASS' if is_win == case['expected'] else '❌ FAIL'}")


def test_tenpai_and_waiting_tiles():
    """テンパイ判定と待ち牌テスト"""
    print("\n\n=== テンパイ判定・待ち牌テスト ===")
    
    game = MahjongGame(GameMode.SIMPLIFIED)
    
    test_cases = [
        {
            "name": "単騎待ち",
            "tiles": ["1萬", "1萬", "1萬", "2萬", "2萬", "2萬", "3萬", "3萬", "3萬", "4萬", "4萬", "4萬", "東"],
            "expected_tenpai": True,
            "expected_waiting": ["東"]
        },
        {
            "name": "両面待ち",
            "tiles": ["1萬", "2萬", "3萬", "4萬", "5萬", "6萬", "7萬", "8萬", "東", "東", "東", "南", "南"],
            "expected_tenpai": True,
            "expected_waiting": ["3萬", "6萬", "9萬"]  # 実際は複数の待ちパターンがある
        },
        {
            "name": "シャンポン待ち",
            "tiles": ["1萬", "1萬", "2萬", "2萬", "3萬", "3萬", "3萬", "4萬", "4萬", "4萬", "5萬", "5萬", "6萬"],
            "expected_tenpai": True,
            "expected_waiting": ["1萬", "2萬", "5萬", "6萬", "7萬"]  # 複数の和了形がある複雑な待ち
        },
        {
            "name": "ノーテン",
            "tiles": ["1萬", "3萬", "5萬", "7萬", "9萬", "東", "南", "西", "北", "白", "發", "中", "2萬"],
            "expected_tenpai": False,
            "expected_waiting": []
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"手牌: {' '.join(case['tiles'])}")
        
        # 手牌作成
        hand = create_test_hand(case['tiles'])
        
        # テンパイ判定
        is_tenpai = game.is_tenpai(hand)
        
        # 待ち牌取得
        waiting_tiles = game.get_waiting_tiles(hand)
        waiting_str = [str(tile) for tile in waiting_tiles]
        
        print(f"テンパイ判定: {'テンパイ' if is_tenpai else 'ノーテン'}")
        print(f"待ち牌: {', '.join(waiting_str) if waiting_str else 'なし'}")
        
        # 期待結果と比較
        tenpai_correct = is_tenpai == case['expected_tenpai']
        waiting_correct = set(waiting_str) == set(case['expected_waiting'])
        
        print(f"期待テンパイ: {'テンパイ' if case['expected_tenpai'] else 'ノーテン'}")
        print(f"期待待ち牌: {', '.join(case['expected_waiting']) if case['expected_waiting'] else 'なし'}")
        print(f"テスト: {'✅ PASS' if tenpai_correct and waiting_correct else '❌ FAIL'}")
        
        if not tenpai_correct:
            print(f"  ❌ テンパイ判定が間違っています")
        if not waiting_correct:
            print(f"  ❌ 待ち牌が間違っています")


def test_ron_win():
    """ロン和了のテスト"""
    print("\n\n=== ロン和了テスト ===")
    
    game = MahjongGame(GameMode.SIMPLIFIED)
    game.start_game()
    
    # プレイヤー1にテンパイ手を設定
    test_tiles = ["1萬", "1萬", "1萬", "2萬", "2萬", "2萬", "3萬", "3萬", "3萬", "4萬", "4萬", "4萬", "東"]
    game.players[0].tiles = []
    for tile_str in test_tiles:
        if tile_str[-1] == '萬':
            number = int(tile_str[0])
            tile = Tile(TileType.MAN, number)
        else:
            tile = Tile(TileType.HONOR, honor=tile_str)
        game.players[0].add_tile(tile)
    
    print("プレイヤー1の手牌:")
    print(' '.join([str(tile) for tile in game.players[0].tiles]))
    
    # テンパイ確認
    is_tenpai = game.is_tenpai(game.players[0])
    waiting_tiles = game.get_waiting_tiles(game.players[0])
    
    print(f"テンパイ: {'はい' if is_tenpai else 'いいえ'}")
    print(f"待ち牌: {', '.join([str(tile) for tile in waiting_tiles])}")
    
    # ロン和了テスト
    test_ron_tiles = [
        Tile(TileType.HONOR, honor="東"),  # 和了牌
        Tile(TileType.MAN, 5),  # 非和了牌
        Tile(TileType.HONOR, honor="南")   # 非和了牌
    ]
    
    print("\nロン和了テスト:")
    for i, ron_tile in enumerate(test_ron_tiles, 1):
        can_ron = game.can_ron(0, ron_tile)
        print(f"{i}. {ron_tile}: {'ロン可能' if can_ron else 'ロン不可'}")
        
        # 実際に和了できるかテスト
        if can_ron:
            test_hand = game.players[0].tiles.copy() + [ron_tile]
            is_actual_win = game._is_winning_hand(test_hand)
            print(f"   実際の和了判定: {'和了' if is_actual_win else '未和了'}")


def test_furiten():
    """フリテンのテスト"""
    print("\n\n=== フリテンテスト ===")
    
    game = MahjongGame(GameMode.SIMPLIFIED)
    game.start_game()
    
    # プレイヤー1にテンパイ手を設定
    test_tiles = ["1萬", "1萬", "1萬", "2萬", "2萬", "2萬", "3萬", "3萬", "3萬", "4萬", "4萬", "4萬", "南"]
    game.players[0].tiles = []
    for tile_str in test_tiles:
        if tile_str[-1] == '萬':
            number = int(tile_str[0])
            tile = Tile(TileType.MAN, number)
        else:
            tile = Tile(TileType.HONOR, honor=tile_str)
        game.players[0].add_tile(tile)
    
    print("プレイヤー1の手牌:")
    print(' '.join([str(tile) for tile in game.players[0].tiles]))
    
    waiting_tiles = game.get_waiting_tiles(game.players[0])
    print(f"待ち牌: {', '.join([str(tile) for tile in waiting_tiles])}")
    
    # フリテンテスト1: 捨て牌なし
    is_furiten = game.is_furiten(0)
    print(f"\n1. 捨て牌なし: フリテン={'はい' if is_furiten else 'いいえ'}")
    
    # フリテンテスト2: 待ち牌を捨てる
    game.players[0].discarded.append(Tile(TileType.HONOR, honor="南"))  # 待ち牌を捨てる
    is_furiten = game.is_furiten(0)
    print(f"2. 南を捨てた後: フリテン={'はい' if is_furiten else 'いいえ'}")
    
    # フリテンテスト3: 待ち牌以外を捨てる
    game.players[0].discarded = [Tile(TileType.MAN, 5)]
    is_furiten = game.is_furiten(0)
    print(f"3. 5萬を捨てた後: フリテン={'はい' if is_furiten else 'いいえ'}")
    
    # ロン判定への影響（フリテン状態で）
    game.players[0].discarded = [Tile(TileType.HONOR, honor="南")]  # 待ち牌を捨てた状態
    ron_tile = Tile(TileType.HONOR, honor="南")
    can_ron_with_furiten = game.can_ron(0, ron_tile)
    print(f"4. フリテン時のロン判定: {'ロン可能' if can_ron_with_furiten else 'ロン不可'}")


if __name__ == "__main__":
    test_winning_hands()
    test_tenpai_and_waiting_tiles()
    test_ron_win()
    test_furiten()