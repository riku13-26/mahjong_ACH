#!/usr/bin/env python3
"""
check_environment.py - 開発環境の確認スクリプト
"""

import sys
import importlib
import subprocess
from pathlib import Path


def check_python_version():
    """Pythonバージョンチェック"""
    print("=== Python環境 ===")
    print(f"Python: {sys.version}")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python 3.8以上")
    else:
        print("❌ Python 3.8以上が必要です")
    print()


def check_packages():
    """必要なパッケージのチェック"""
    print("=== パッケージ確認 ===")
    
    packages = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'gym': 'OpenAI Gym',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'tqdm': 'tqdm',
        'wandb': 'Weights & Biases',
        'hydra': 'Hydra',
        'omegaconf': 'OmegaConf',
        'jupyter': 'Jupyter',
        'pytest': 'pytest'
    }
    
    missing = []
    
    for package, name in packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: Not installed")
            missing.append(package)
    
    if missing:
        print(f"\n以下のパッケージをインストールしてください:")
        print(f"pip install {' '.join(missing)}")
    
    print()


def check_gpu():
    """GPU環境のチェック"""
    print("=== GPU環境 ===")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
        else:
            print("❌ GPU not available (CPU mode)")
    except ImportError:
        print("❌ PyTorchがインストールされていません")
    print()


def check_project_structure():
    """プロジェクト構造のチェック"""
    print("=== プロジェクト構造 ===")
    
    required_dirs = [
        'src',
        'src/envs',
        'src/algorithms',
        'src/utils',
        'tests',
        'configs',
        'notebooks',
        'results',
        'docs'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (missing)")
    
    print()


def check_mahjong_files():
    """麻雀関連ファイルのチェック"""
    print("=== 麻雀ゲームファイル ===")
    
    files = [
        'mahjong_game.py',
        'mahjong_tile.py',
        'test_game.py',
        'requirements.txt'
    ]
    
    for file in files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} (not found)")
    
    print()


def test_mahjong_import():
    """麻雀モジュールのインポートテスト"""
    print("=== インポートテスト ===")
    
    try:
        from mahjong_game import MahjongGame, GameMode
        from mahjong_tile import Tile, TileType, TileDeck, Hand
        print("✅ 麻雀モジュールのインポート成功")
        
        # 簡単な動作テスト
        game = MahjongGame(GameMode.SIMPLIFIED)
        game.start_game()
        state = game.get_game_state()
        print(f"✅ ゲーム初期化成功 (山牌: {state['deck_remaining']}枚)")
        
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
    
    print()


def check_git():
    """Gitリポジトリのチェック"""
    print("=== Git環境 ===")
    
    try:
        # Gitがインストールされているか
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        
        # Gitリポジトリか確認
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Gitリポジトリ初期化済み")
            
            # 現在のブランチ
            result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ 現在のブランチ: {result.stdout.strip()}")
        else:
            print("❌ Gitリポジトリが初期化されていません")
            print("   実行: git init")
    except FileNotFoundError:
        print("❌ Gitがインストールされていません")
    
    print()


def main():
    """メイン実行"""
    print("=" * 60)
    print("麻雀AI開発環境チェック")
    print("=" * 60)
    print()
    
    check_python_version()
    check_packages()
    check_gpu()
    check_project_structure()
    check_mahjong_files()
    test_mahjong_import()
    check_git()
    
    print("=" * 60)
    print("チェック完了")
    print("=" * 60)


if __name__ == "__main__":
    main()