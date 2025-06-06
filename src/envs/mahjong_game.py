from typing import List, Dict, Optional, Tuple
from .mahjong_tile import Tile, TileDeck, Hand, TileType
import random
from enum import Enum


class GameMode(Enum):
    NORMAL = "normal"
    SIMPLIFIED = "simplified"


class GameResult(Enum):
    WIN = "win"
    LOSE = "lose"
    DRAW_TENPAI = "draw_tenpai"
    DRAW_NOTEN = "draw_noten"
    RIICHI = "riichi"


class MahjongGame:
    def __init__(self, mode: GameMode = GameMode.NORMAL):
        self.mode = mode
        self.deck = TileDeck(simplified=(mode == GameMode.SIMPLIFIED))
        allow_melds = (mode == GameMode.NORMAL)  # 簡素化モードでは鳴きなし
        self.players = [Hand(allow_melds=allow_melds), Hand(allow_melds=allow_melds)]
        self.current_player = 0
        self.game_state = "waiting"
        self.dora_indicators = []
        self.round_wind = "東"
        self.player_winds = ["東", "南"]
        self.riichi_declared = [False, False]
        self.rewards = [0.0, 0.0]
        
    def start_game(self):
        self.deck.shuffle()
        self._deal_initial_hands()
        self._set_dora()
        self.game_state = "playing"
        
    def _deal_initial_hands(self):
        # 二人麻雀では各プレイヤー13枚ずつ配る
        for player in self.players:
            for _ in range(13):
                player.add_tile(self.deck.draw())
        
        # 最初のプレイヤーに1枚追加（14枚でスタート）
        self.players[self.current_player].add_tile(self.deck.draw())
    
    def _set_dora(self):
        # ドラ表示牌を1枚表示
        if self.deck.remaining_count() > 0:
            self.dora_indicators.append(self.deck.draw())
    
    def get_current_player(self) -> Hand:
        return self.players[self.current_player]
    
    def discard_tile(self, tile: Tile) -> bool:
        current_hand = self.get_current_player()
        if current_hand.discard_tile(tile):
            self._switch_player()
            return True
        return False
    
    def draw_tile(self) -> Optional[Tile]:
        if self.deck.remaining_count() > 0:
            tile = self.deck.draw()
            self.get_current_player().add_tile(tile)
            return tile
        return None
    
    def _switch_player(self):
        self.current_player = (self.current_player + 1) % 2
        
        # 次のプレイヤーがツモ
        drawn_tile = self.draw_tile()
        return drawn_tile
    
    def check_win(self, hand: Hand) -> bool:
        return self._is_winning_hand(hand.tiles + hand.melds)
    
    def _is_winning_hand(self, tiles: List[Tile]) -> bool:
        # 簡単な和了判定（14枚必要）
        if len(tiles) != 14:
            return False
        
        tile_counts = {}
        for tile in tiles:
            tile_counts[tile] = tile_counts.get(tile, 0) + 1
        
        # 七対子チェック
        if self._check_seven_pairs(tile_counts):
            return True
        
        # 一般的な和了形チェック
        return self._check_standard_win(tile_counts)
    
    def _check_seven_pairs(self, tile_counts: Dict[Tile, int]) -> bool:
        pairs = 0
        for count in tile_counts.values():
            if count == 2:
                pairs += 1
            elif count != 0:
                return False
        return pairs == 7
    
    def _check_standard_win(self, tile_counts: Dict[Tile, int]) -> bool:
        # 雀頭（対子）を見つける
        for tile, count in tile_counts.items():
            if count >= 2:
                # 雀頭を仮定して残りをチェック
                temp_counts = tile_counts.copy()
                temp_counts[tile] -= 2
                if temp_counts[tile] == 0:
                    del temp_counts[tile]
                
                if self._check_melds(temp_counts):
                    return True
        return False
    
    def _check_melds(self, tile_counts: Dict[Tile, int]) -> bool:
        # 残り12枚が面子（刻子・順子）で構成されているかチェック
        if not tile_counts:
            return True
        
        # 刻子チェック
        for tile, count in list(tile_counts.items()):
            if count >= 3:
                temp_counts = tile_counts.copy()
                temp_counts[tile] -= 3
                if temp_counts[tile] == 0:
                    del temp_counts[tile]
                
                if self._check_melds(temp_counts):
                    return True
        
        # 順子チェック（数牌のみ）
        for tile, count in list(tile_counts.items()):
            if (tile.tile_type != TileType.HONOR and 
                tile.number <= 7 and count > 0):
                
                tile2 = Tile(tile.tile_type, tile.number + 1)
                tile3 = Tile(tile.tile_type, tile.number + 2)
                
                if (tile2 in tile_counts and tile3 in tile_counts and
                    tile_counts[tile2] > 0 and tile_counts[tile3] > 0):
                    
                    temp_counts = tile_counts.copy()
                    temp_counts[tile] -= 1
                    temp_counts[tile2] -= 1
                    temp_counts[tile3] -= 1
                    
                    for t in [tile, tile2, tile3]:
                        if temp_counts[t] == 0:
                            del temp_counts[t]
                    
                    if self._check_melds(temp_counts):
                        return True
        
        return False
    
    def get_game_state(self) -> Dict:
        return {
            "mode": self.mode.value,
            "current_player": self.current_player,
            "game_state": self.game_state,
            "deck_remaining": self.deck.remaining_count(),
            "dora_indicators": [str(tile) for tile in self.dora_indicators],
            "riichi_declared": self.riichi_declared.copy(),
            "rewards": self.rewards.copy(),
            "player_hands": [
                {
                    "tiles": [str(tile) for tile in player.tiles],
                    "tile_count": player.count_tiles(),
                    "discarded": [str(tile) for tile in player.discarded],
                    "melds": [[str(tile) for tile in meld] for meld in player.melds],
                    "allow_melds": player.allow_melds
                }
                for player in self.players
            ]
        }
    
    def is_game_over(self) -> bool:
        # ゲーム終了条件をチェック
        for player in self.players:
            if self.check_win(player):
                return True
        
        # 牌が尽きた場合
        if self.deck.remaining_count() == 0:
            return True
        
        return False
    
    def declare_riichi(self, player_index: int) -> bool:
        """リーチ宣言"""
        if self.riichi_declared[player_index]:
            return False
        
        # リーチ可能かチェック（テンパイ状態）
        if self.is_tenpai(self.players[player_index]):
            self.riichi_declared[player_index] = True
            self.players[player_index].riichi_declared = True
            self.rewards[player_index] -= 0.1  # リーチ料
            return True
        return False
    
    def is_tenpai(self, hand: Hand) -> bool:
        """テンパイ判定"""
        tiles = hand.tiles.copy()
        
        # 全ての可能な牌を試して、和了になるかチェック
        all_tiles = self._get_all_possible_tiles()
        for tile in all_tiles:
            test_tiles = tiles + [tile]
            if self._is_winning_hand(test_tiles):
                return True
        return False
    
    def _get_all_possible_tiles(self) -> List[Tile]:
        """全ての可能な牌を取得"""
        tiles = []
        
        if self.mode == GameMode.SIMPLIFIED:
            # 萬子のみ
            for number in range(1, 10):
                tiles.append(Tile(TileType.MAN, number))
            # 三元牌・風牌
            for honor in ["東", "南", "西", "北", "白", "發", "中"]:
                tiles.append(Tile(TileType.HONOR, honor=honor))
        else:
            # 通常モード：全ての牌
            for tile_type in [TileType.MAN, TileType.PIN, TileType.SOU]:
                for number in range(1, 10):
                    tiles.append(Tile(tile_type, number))
            for honor in ["東", "南", "西", "北", "白", "發", "中"]:
                tiles.append(Tile(TileType.HONOR, honor=honor))
        
        return tiles
    
    def calculate_reward(self, player_index: int, result: GameResult) -> float:
        """報酬計算"""
        reward = 0.0
        
        if result == GameResult.WIN:
            reward = 1.0
            # リーチしていた場合は返還
            if self.riichi_declared[player_index]:
                reward += 0.2
        elif result == GameResult.LOSE:
            reward = -1.0
        elif result == GameResult.DRAW_TENPAI:
            reward = 0.2
        elif result == GameResult.DRAW_NOTEN:
            reward = 0.0
        
        return reward
    
    def end_game(self) -> Dict[str, float]:
        """ゲーム終了時の報酬計算"""
        results = {}
        
        # 和了チェック
        winner = None
        for i, player in enumerate(self.players):
            if self.check_win(player):
                winner = i
                break
        
        if winner is not None:
            # 和了
            for i in range(len(self.players)):
                if i == winner:
                    self.rewards[i] += self.calculate_reward(i, GameResult.WIN)
                else:
                    self.rewards[i] += self.calculate_reward(i, GameResult.LOSE)
        else:
            # 流局
            for i, player in enumerate(self.players):
                if self.is_tenpai(player):
                    self.rewards[i] += self.calculate_reward(i, GameResult.DRAW_TENPAI)
                else:
                    self.rewards[i] += self.calculate_reward(i, GameResult.DRAW_NOTEN)
        
        return {f"player_{i}": self.rewards[i] for i in range(len(self.players))}
    
    def get_waiting_tiles(self, hand: Hand) -> List[Tile]:
        """待ち牌を取得"""
        waiting_tiles = []
        tiles = hand.tiles.copy()
        
        all_tiles = self._get_all_possible_tiles()
        for tile in all_tiles:
            test_tiles = tiles + [tile]
            if self._is_winning_hand(test_tiles):
                # 重複を避けるため、既に追加されていない場合のみ追加
                if not any(existing_tile == tile for existing_tile in waiting_tiles):
                    waiting_tiles.append(tile)
        
        return waiting_tiles
    
    def is_furiten(self, player_index: int) -> bool:
        """フリテン判定"""
        hand = self.players[player_index]
        waiting_tiles = self.get_waiting_tiles(hand)
        
        # 自分の捨て牌に待ち牌があるかチェック
        for waiting_tile in waiting_tiles:
            for discarded_tile in hand.discarded:
                if waiting_tile == discarded_tile:
                    return True
        return False
    
    def can_ron(self, player_index: int, tile: Tile) -> bool:
        """ロン和了可能かチェック"""
        hand = self.players[player_index]
        
        # フリテンチェック
        if self.is_furiten(player_index):
            return False
        
        # 和了牌かチェック
        test_tiles = hand.tiles + [tile]
        return self._is_winning_hand(test_tiles)
    
    def ron_win(self, player_index: int, tile: Tile) -> bool:
        """ロン和了処理"""
        if not self.can_ron(player_index, tile):
            return False
        
        # 手牌に和了牌を追加
        self.players[player_index].add_tile(tile)
        self.game_state = "finished"
        return True