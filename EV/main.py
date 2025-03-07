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

import os
import sys
import argparse
import datetime
import gymnasium as gym
import tensorflow as tf
import numpy as np
import warnings


def main(play, epochs, pop_size, sigma, lr, silent=False):
    """メインエントリポイント関数（学習または実行）
    
    Args:
        play (bool): Trueの場合は既存モデルでプレイ、Falseの場合は学習を実行
        epochs (int): 学習エポック数
        pop_size (int): 集団サイズ
        sigma (float): ノイズ幅
        lr (float): 学習率
        silent (bool): 警告・情報メッセージを抑制するかどうか
    """
    from config import Config
    from agent import EvolutionalAgent
    from environment import CartPoleVectorObserver
    from trainer import EvolutionalTrainer
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
    
    if play:
        # "human"を指定して可視化環境を作成
        env = CartPoleVectorObserver(render_mode="human")
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    else:
        # トレーナー初期化
        trainer = EvolutionalTrainer(
            population_size=pop_size,  # 集団サイズ
            sigma=sigma,               # ノイズ幅
            learning_rate=lr,          # 学習率
            report_interval=5          # ログ間隔
        )
        # 学習実行
        trained = trainer.train(
            epoch=epochs,             # エポック数
            episode_per_agent=Config.DEFAULT_EPISODE_PER_AGENT,  # 1個体あたりのエピソード数
            silent=silent             # サイレントフラグ
        )
        trained.save(model_path)      # モデル保存
        
        # 結果表示
        if not silent:
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
        
        # 結果データから適切なy軸最大値を判断
        max_reward = max([rs.mean() for rs in trainer.reward_log])
        y_max = None  # 自動設定をデフォルトに
        
        if max_reward < 100:
            y_max = 120  # 低い報酬時は細かく表示
        elif max_reward < 300:
            y_max = 350  # 中程度の報酬時
            
        # 結果の保存と表示
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        result_filename = f"es_results_e{epochs}_p{pop_size}_s{sigma}_lr{lr}_{timestamp}.png"
        trainer.plot_rewards(result_filename, y_max=y_max)


# スクリプト実行時のエントリーポイント
if __name__ == "__main__":
    from config import Config
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
