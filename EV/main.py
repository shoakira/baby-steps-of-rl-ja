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

# 内部モジュール（早期インポートを避けるため関数内でインポート）


def setup_environment(silent: bool = False) -> Tuple[datetime.datetime, str]:
    """環境セットアップと初期化を行う
    
    Args:
        silent: 出力を抑制するかどうか
        
    Returns:
        開始時刻とモデルパス
    """
    from config import Config
    from utils import configure_tensorflow
    
    start_time = datetime.datetime.now()
    
    # サイレントモード設定
    if silent:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        warnings.filterwarnings('ignore')
    else:
        print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # TensorFlow設定（GPUの初期化など）
    configure_tensorflow(silent=silent)
    
    # モデル保存パス設定
    model_path = Config.get_model_path()
    
    return start_time, model_path


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
    silent: bool = False
) -> Dict[str, Any]:
    """エージェントを学習し、結果を返す
    
    Args:
        epochs: 学習エポック数
        pop_size: 集団サイズ
        sigma: ノイズ幅
        lr: 学習率
        silent: サイレントフラグ
        
    Returns:
        学習結果の辞書（トレーナー、開始時刻など）
    """
    from config import Config
    from agent import EvolutionalAgent
    from trainer import EvolutionalTrainer
    
    start_time, model_path = setup_environment(silent)
    
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
    
    # メモリ使用状況を表示（オプション）
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"メモリ使用量: {memory_info.rss / (1024 * 1024):.1f} MB")
    except ImportError:
        # psutilがインストールされていない場合は表示しない
        pass
    
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
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            epochs = len(reward_data)
            pop_size = trainer.population_size
            sigma = trainer.sigma
            lr = trainer.learning_rate
            
            result_filename = f"es_results_e{epochs}_p{pop_size}_s{sigma}_lr{lr}_{timestamp}.png"
            trainer.plot_rewards(result_filename, y_max=y_max)
    except Exception as e:
        print(f"警告: 結果の表示中にエラーが発生しました: {e}")


def main(play: bool, epochs: int, pop_size: int, sigma: float, lr: float, silent: bool = False) -> None:
    """メインエントリポイント関数
    
    Args:
        play: Trueの場合は既存モデルでプレイ、Falseの場合は学習を実行
        epochs: 学習エポック数
        pop_size: 集団サイズ
        sigma: ノイズ幅
        lr: 学習率
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
            results = train_agent(epochs, pop_size, sigma, lr, silent)
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
    parser.add_argument("--silent", action="store_true", help="警告を抑制")

    # 引数解析
    args = parser.parse_args()
    
    # メイン処理実行
    main(args.play, args.epochs, args.pop_size, args.sigma, args.lr, args.silent)
