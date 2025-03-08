#!/usr/bin/env python
"""タイムアウト問題のデバッグ用スクリプト"""

import os
import sys
import time
import psutil
from datetime import datetime

# カレントディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from agent import EvolutionalAgent
from environment import CartPoleVectorObserver

# デバッグ設定
Config.DEFAULT_POPULATION_SIZE = 10  # 少ない個体数
Config.DEFAULT_EPOCHS = 5            # 少ないエポック
Config.EVALUATION_TIMEOUT_SECONDS = 30  # 短いタイムアウト
Config.DEBUG_MODEL_SUMMARY = True    # モデルサマリー表示

# デバッグログファイル
log_file = os.path.join(current_dir, "debug_log.txt")

def log(message):
    """ログの出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")

def main():
    """デバッグ実行"""
    # ログファイル初期化
    with open(log_file, "w") as f:
        f.write(f"===== デバッグ開始: {datetime.now()} =====\n")
    
    # TensorFlowを強制的にCPUモードで実行
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # すべてのログを表示
    
    # 1. 単一エージェントのテスト
    log("単一エージェントのテスト開始")
    test_single_agent()
    
    # 2. 並列処理のテスト
    log("並列処理のテスト開始")
    test_parallel()

def test_single_agent():
    """単一エージェントのテスト"""
    env = CartPoleVectorObserver()
    agent = EvolutionalAgent(list(range(env.action_space.n)))
    
    # 初期化のテスト
    log("エージェント初期化開始")
    s = env.reset()
    agent.initialize(s)
    log("エージェント初期化完了")
    
    # 推論のテスト
    log("推論テスト開始")
    for i in range(10):
        start = time.time()
        action = agent.policy(s)
        n_state, reward, done, info = env.step(action)
        s = n_state
        log(f"ステップ {i}: 行動={action}, 報酬={reward}, 処理時間={time.time()-start:.6f}秒")
    log("推論テスト完了")

def test_parallel():
    """並列処理のテスト"""
    from trainer import EvolutionalTrainer
    
    log("トレーナー初期化")
    trainer = EvolutionalTrainer(
        population_size=4,  # 少数のプロセスでテスト
        sigma=0.1,
        learning_rate=0.05,
        report_interval=1,
        timeout_seconds=15  # 短いタイムアウト
    )
    
    log("学習開始（限定的なエポック数）")
    monitor_thread = start_monitoring()
    
    try:
        # 並列処理監視しながら学習実行
        trainer.train(epoch=2, episode_per_agent=2)
        log("学習正常終了")
    except Exception as e:
        log(f"学習中にエラー発生: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        stop_monitoring(monitor_thread)

def start_monitoring():
    """プロセス監視スレッドの開始"""
    import threading
    stop_flag = threading.Event()
    
    def monitor_processes():
        while not stop_flag.is_set():
            try:
                # 現在のプロセスの子プロセスを取得
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
                
                # プロセス情報をログに出力
                if children:
                    log(f"実行中の子プロセス数: {len(children)}")
                    for i, p in enumerate(children):
                        try:
                            cpu_percent = p.cpu_percent(interval=0.1)
                            memory_info = p.memory_info()
                            memory_mb = memory_info.rss / (1024 * 1024)
                            log(f"  子プロセス {i}: PID={p.pid}, CPU={cpu_percent:.1f}%, "
                                f"メモリ={memory_mb:.1f}MB, 状態={p.status()}")
                        except:
                            pass
            except Exception as e:
                log(f"監視エラー: {e}")
            
            time.sleep(2.0)
    
    # スレッド開始
    thread = threading.Thread(target=monitor_processes, daemon=True)
    thread.start()
    log("プロセス監視スレッド開始")
    return (thread, stop_flag)

def stop_monitoring(monitor_data):
    """監視スレッドの停止"""
    thread, stop_flag = monitor_data
    stop_flag.set()
    thread.join(timeout=1.0)
    log("プロセス監視スレッド停止")

if __name__ == "__main__":
    main()