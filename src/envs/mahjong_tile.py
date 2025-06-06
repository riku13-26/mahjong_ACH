from enum import Enum
from typing import List, Dict, Set


class TileType(Enum):
    MAN = "萬子"
    PIN = "筒子"
    SOU = "索子"
    HONOR = "字牌"


class Tile:
    def __init__(self, tile_type: TileType, number: int = None, honor: str = None):
        self.tile_type = tile_type
        self.number = number
        self.honor = honor
        
        if tile_type == TileType.HONOR:
            if honor not in ["東", "南", "西", "北", "白", "發", "中"]:
                raise ValueError("Invalid honor tile")
        else:
            if number not in range(1, 10):
                raise ValueError("Number must be between 1 and 9")
    
    def __str__(self):
        if self.tile_type == TileType.HONOR:
            return self.honor
        else:
            return f"{self.number}{self.tile_type.value[0]}"
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return (self.tile_type == other.tile_type and 
                self.number == other.number and 
                self.honor == other.honor)
    
    def __hash__(self):
        return hash((self.tile_type, self.number, self.honor))


class TileDeck:
    def __init__(self, simplified: bool = False):
        self.simplified = simplified
        self.tiles = self._create_deck()
        self.discarded = []
    
    def _create_deck(self) -> List[Tile]:
        tiles = []
        
        if self.simplified:
            # 簡素化モード：萬子のみ
            for number in range(1, 10):
                for _ in range(4):
                    tiles.append(Tile(TileType.MAN, number))
        else:
            # 通常モード：萬子、筒子、索子 (1-9) × 4枚ずつ
            for tile_type in [TileType.MAN, TileType.PIN, TileType.SOU]:
                for number in range(1, 10):
                    for _ in range(4):
                        tiles.append(Tile(tile_type, number))
        
        # 字牌（三元牌・風牌） × 4枚ずつ
        for honor in ["東", "南", "西", "北", "白", "發", "中"]:
            for _ in range(4):
                tiles.append(Tile(TileType.HONOR, honor=honor))
        
        return tiles
    
    def shuffle(self):
        import random
        random.shuffle(self.tiles)
    
    def draw(self) -> Tile:
        if not self.tiles:
            raise ValueError("No more tiles to draw")
        return self.tiles.pop()
    
    def remaining_count(self) -> int:
        return len(self.tiles)


class Hand:
    def __init__(self, allow_melds: bool = True):
        self.tiles: List[Tile] = []
        self.melds: List[List[Tile]] = []
        self.discarded: List[Tile] = []
        self.allow_melds = allow_melds
        self.riichi_declared = False
    
    def add_tile(self, tile: Tile):
        self.tiles.append(tile)
        self.sort_hand()
    
    def call_meld(self, tiles: List[Tile]) -> bool:
        """鳴きを行う"""
        if not self.can_call_meld(tiles):
            return False
        
        # 手牌から牌を除去
        for tile in tiles:
            if not self.remove_tile(tile):
                return False
        
        # 面子に追加
        self.melds.append(tiles)
        return True
    
    def remove_tile(self, tile: Tile) -> bool:
        if tile in self.tiles:
            self.tiles.remove(tile)
            return True
        return False
    
    def discard_tile(self, tile: Tile) -> bool:
        if self.remove_tile(tile):
            self.discarded.append(tile)
            return True
        return False
    
    def sort_hand(self):
        def tile_sort_key(tile):
            type_order = {TileType.MAN: 0, TileType.PIN: 1, TileType.SOU: 2, TileType.HONOR: 3}
            if tile.tile_type == TileType.HONOR:
                honor_order = {"東": 1, "南": 2, "西": 3, "北": 4, "白": 5, "發": 6, "中": 7}
                return (type_order[tile.tile_type], honor_order[tile.honor])
            else:
                return (type_order[tile.tile_type], tile.number)
        
        self.tiles.sort(key=tile_sort_key)
    
    def count_tiles(self) -> int:
        return len(self.tiles)
    
    def get_tile_counts(self) -> Dict[Tile, int]:
        counts = {}
        for tile in self.tiles:
            counts[tile] = counts.get(tile, 0) + 1
        return counts
    
    def can_call_meld(self, tiles: List[Tile]) -> bool:
        """鳴き可能かチェック"""
        if not self.allow_melds:
            return False
        
        if len(tiles) not in [3, 4]:  # ポン・チー・カンのみ
            return False
        
        # 手牌に必要な牌があるかチェック
        hand_counts = self.get_tile_counts()
        meld_counts = {}
        for tile in tiles:
            meld_counts[tile] = meld_counts.get(tile, 0) + 1
        
        for tile, count in meld_counts.items():
            if hand_counts.get(tile, 0) < count:
                return False
        
        return True