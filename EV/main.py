"""
# 進化戦略（Evolution Strategy）強化学習アルゴリズム
#
# ファイル構成:
# - config.py: 設定と定数
# - agent.py: エージェントの実装
# - environment.py: 環境ラッパー
# - trainer.py: 進化戦略トレーナー
# - utils.py: ユーティリティ関数
# - main.py: メインエントリポイント
"""
###############################################################################
# main.py - メインエントリポイント
###############################################################################

# 標準ライブラリ
import os
import sys
import time
import argparse
import datetime
import warnings
from typing import Optional, Tuple, Dict, Any

# サードパーティライブラリ
import numpy as np
import gymnasium as gym
import tensorflow as tf


def setup_silent_mode(silent: bool) -> None:
    """サイレントモードの設定
    
    警告やログ出力を抑制します
    
    Args:
        silent: 出力を抑制するかどうか
    """
    if silent:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings('ignore')


def setup_tensorflow(silent: bool) -> None:
    """TensorFlow環境の設定
    
    GPUの初期化とメモリ設定を行います
    
    Args:
        silent: 出力を抑制するかどうか
    """
    from utils import configure_tensorflow
    configure_tensorflow(silent=silent)


def get_model_path() -> str:
    """モデル保存パスを取得
    
    Returns:
        str: モデル保存先のフルパス
    """
    from config import Config
    return Config.get_model_path()


def setup_environment(silent: bool = False) -> Tuple[datetime.datetime, str]:
    """環境セットアップと初期化を行う
    
    Args:
        silent: 出力を抑制するかどうか
        
    Returns:
        開始時刻とモデルパス
    """
    start_time = datetime.datetime.now()
    
    # サイレントモード設定
    setup_silent_mode(silent)
    
    if not silent:
        print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # TensorFlow設定
    setup_tensorflow(silent)
    
    # モデル保存パス設定
    model_path = get_model_path()
    
    return start_time, model_path


def _generate_result_filename(trainer, timestamp=None) -> str:
    """学習結果のファイル名を生成
    
    Args:
        trainer: トレーナーオブジェクト
        timestamp: タイムスタンプ（Noneの場合は現在時刻）
        
    Returns:
        str: 結果ファイル名
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
    epochs = len(trainer.reward_log)
    pop_size = trainer.population_size
    sigma = trainer.sigma
    lr = trainer.learning_rate
    
    return f"es_results_e{epochs}_p{pop_size}_s{sigma}_lr{lr}_{timestamp}.png"


def get_memory_usage() -> Optional[Tuple[float, str]]:
    """現在のメモリ使用量を取得
    
    Returns:
        Optional[Tuple[float, str]]: (使用メモリMB, フォーマット済み文字列) またはNone
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        return memory_mb, f"メモリ使用量: {memory_mb:.1f} MB"
    except ImportError:
        return None


def play_agent(model_path: str) -> None:
    """学習済みエージェントで環境を実行
    
    Args:
        model_path: 学習済みモデルのパス
    """
    from environment import CartPoleVectorObserver
    from agent import EvolutionalAgent
    
    try:
        # "human"を指定して可視化環境を作成
        env = CartPoleVectorObserver(render_mode="human")
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    except Exception as e:
        print(f"エラー: エージェントの実行に失敗しました: {e}")
        sys.exit(1)


def train_agent(
    epochs: int, 
    pop_size: int, 
    sigma: float, 
    lr: float, 
    cores: Optional[int] = None,
    force_cpu: bool = False,
    silent: bool = False
) -> Dict[str, Any]:
    """エージェントを学習し、結果を返す
    
    Args:
        epochs: 学習エポック数
        pop_size: 集団サイズ
        sigma: ノイズ幅
        lr: 学習率
        cores: 使用するCPUコア数（Noneの場合は自動）
        force_cpu: CPUのみを強制使用するフラグ
        silent: サイレントフラグ
        
    Returns:
        学習結果の辞書（トレーナー、開始時刻など）
    """
    from config import Config
    from agent import EvolutionalAgent
    from trainer import EvolutionalTrainer
    
    # 環境セットアップ
    start_time, model_path = setup_environment(silent)
    
    # CPU専用モード設定（オプション）
    if force_cpu:
        Config.force_cpu_mode()
        
    # コア数設定（オプション）
    if cores is not None:
        Config.OVERRIDE_CORES = cores
    
    try:
        # トレーナー初期化
        trainer = EvolutionalTrainer(
            population_size=pop_size,
            sigma=sigma,
            learning_rate=lr,
            report_interval=5
        )
        
        # 学習実行
        trained = trainer.train(
            epoch=epochs,
            episode_per_agent=Config.DEFAULT_EPISODE_PER_AGENT, 
            silent=silent
        )
        
        # モデル保存
        trained.save(model_path)
        
        return {
            "trainer": trainer,
            "model_path": model_path,
            "start_time": start_time,
            "trained_agent": trained
        }
        
    except KeyboardInterrupt:
        print("\n学習が中断されました。これまでの結果を保存します...")
        if 'trained' in locals():
            trained.save(f"{model_path}.interrupted")
            print(f"中断された学習結果を保存: {model_path}.interrupted")
        return {}
        
    except Exception as e:
        print(f"エラー: 学習中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return {}


def display_results(results: Dict[str, Any], silent: bool = False) -> None:
    """学習結果を表示・保存する
    
    Args:
        results: 学習結果の辞書
        silent: サイレントフラグ
    """
    if not results or silent:
        return
        
    trainer = results.get("trainer")
    model_path = results.get("model_path")
    start_time = results.get("start_time")
    
    if not all([trainer, model_path, start_time]):
        return
    
    # 実行時間計算
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print(f"総実行時間: {elapsed.total_seconds():.1f}秒")
    print(f"モデル保存先: {model_path}")
    
    # メモリ使用状況を表示
    memory_info = get_memory_usage()
    if memory_info:
        print(memory_info[1])
    
    # 結果の保存と表示
    try:
        # 結果データから適切なy軸最大値を判断
        reward_data = trainer.reward_log
        if reward_data:
            max_reward = max([rs.mean() for rs in reward_data])
            y_max = None  # 自動設定をデフォルトに
            
            if max_reward < 100:
                y_max = 120  # 低い報酬時は細かく表示
            elif max_reward < 300:
                y_max = 350  # 中程度の報酬時
            
            # 一意のファイル名で結果を保存
            result_filename = _generate_result_filename(trainer)
            trainer.plot_rewards(result_filename, y_max=y_max)
    except Exception as e:
        print(f"警告: 結果の表示中にエラーが発生しました: {e}")


def main(
    play: bool, 
    epochs: int, 
    pop_size: int, 
    sigma: float, 
    lr: float, 
    cores: Optional[int] = None,
    force_cpu: bool = False,
    silent: bool = False
) -> None:
    """メインエントリポイント関数
    
    Args:
        play: Trueの場合は既存モデルでプレイ、Falseの場合は学習を実行
        epochs: 学習エポック数
        pop_size: 集団サイズ
        sigma: ノイズ幅
        lr: 学習率
        cores: 使用するCPUコア数
        force_cpu: CPU専用モードフラグ
        silent: サイレントフラグ
    """
    try:
        # コードが属するパスをシステムパスに追加（インポート問題を回避）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        # 設定のロード（モジュールの相互依存を解決するため遅延インポート）
        from config import Config

        if play:
            # プレイモード
            _, model_path = setup_environment(silent)
            play_agent(model_path)
        else:
            # 学習モード
            results = train_agent(epochs, pop_size, sigma, lr, cores, force_cpu, silent)
            display_results(results, silent)
            
    except Exception as e:
        print(f"致命的エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# スクリプト実行時のエントリーポイント
if __name__ == "__main__":
    # パスの設定
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # 設定モジュールのインポート
    try:
        from config import Config
    except ImportError:
        print("エラー: config モジュールが見つかりません。")
        print("必要なファイル構成:")
        print("- config.py: 設定と定数")
        print("- agent.py: エージェントの実装")
        print("- environment.py: 環境ラッパー")
        print("- trainer.py: 進化戦略トレーナー")
        print("- utils.py: ユーティリティ関数")
        sys.exit(1)
        
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Evolution Strategy Agent")
    parser.add_argument("--play", action="store_true", help="学習済みモデルで実行")
    parser.add_argument("--epochs", type=int, default=Config.DEFAULT_EPOCHS, help="学習エポック数")
    parser.add_argument("--pop-size", type=int, default=Config.DEFAULT_POPULATION_SIZE, help="集団サイズ")
    parser.add_argument("--sigma", type=float, default=Config.DEFAULT_SIGMA, help="探索のノイズ幅")
    parser.add_argument("--lr", type=float, default=Config.DEFAULT_LEARNING_RATE, help="学習率")
    parser.add_argument("--cores", type=int, default=None, help="使用するCPUコア数")
    parser.add_argument("--force-cpu", action="store_true", help="CPU専用モードを強制")
    parser.add_argument("--silent", action="store_true", help="出力を抑制")

    # 引数解析
    args = parser.parse_args()
    
    # メイン処理実行
    main(
        args.play, 
        args.epochs, 
        args.pop_size, 
        args.sigma, 
        args.lr, 
        args.cores, 
        args.force_cpu, 
        args.silent
    )
