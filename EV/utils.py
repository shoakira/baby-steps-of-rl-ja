###############################################################################
# utils.py - ユーティリティ関数
###############################################################################

import os
import sys
import platform
from typing import List, Tuple, Optional

import tensorflow as tf
import numpy as np


def configure_tensorflow(silent: bool = False) -> Tuple[List, bool]:
    """TensorFlow実行環境を最適化
    
    Apple Silicon、CUDA GPUを検出し、環境に応じた最適な設定を行います。
    メモリ使用、マルチスレッド、パフォーマンス関連の設定を調整します。
    
    Args:
        silent: 出力抑制フラグ（Trueの場合、出力を表示しない）
        
    Returns:
        (検出されたGPUリスト, GPU使用フラグ)
    """
    # 設定をインポート（循環参照を防ぐため関数内でインポート）
    from config import Config
    
    # 環境変数設定（ログレベル、スレッド数）
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = Config.TF_LOG_LEVEL
    os.environ["VECLIB_MAXIMUM_THREADS"] = Config.MAX_THREADS
    
    # GPUの検出と設定
    gpus = tf.config.list_physical_devices('GPU')
    is_using_gpu = len(gpus) > 0
    
    if is_using_gpu:
        _configure_gpu_memory(gpus, silent)
    
    # プラットフォーム固有の設定
    if platform.processor() == 'arm':  # Apple Silicon検出
        _configure_apple_silicon(gpus, silent)
    else:
        _configure_standard_platform(gpus, silent)
        
    return gpus, is_using_gpu


def _configure_gpu_memory(gpus: List, silent: bool = False) -> None:
    """GPU メモリ設定の最適化
    
    Args:
        gpus: 検出されたGPUのリスト
        silent: 出力抑制フラグ
    """
    try:
        # すべてのGPUでメモリ動的割り当てを有効化
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        if not silent:
            print(f"検出されたGPU: {len(gpus)}個")
            print("GPU メモリの動的割り当てを有効化しました")
    except RuntimeError as e:
        if not silent:
            print(f"GPU設定エラー: {e}")


def _configure_apple_silicon(gpus: List, silent: bool = False) -> None:
    """Apple Silicon (M1/M2/M3) 固有の設定
    
    Args:
        gpus: 検出されたGPUのリスト
        silent: 出力抑制フラグ
    """
    # oneDNNを無効化（MPS互換性向上）
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # GPU検出時の出力
    if gpus and not silent:
        print("MPS/GPU 加速が有効化されました")
    elif not silent:
        print("CPU モードで実行します")
        
    # マルチスレッド設定（Apple Siliconの8コア活用）
    # - intra_op: 1つの演算内での並列スレッド数
    # - inter_op: 演算グラフ内での並列実行演算数
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def _configure_standard_platform(gpus: List, silent: bool = False) -> None:
    """標準プラットフォーム（x86/AMD64など）向けの設定
    
    Args:
        gpus: 検出されたGPUのリスト
        silent: 出力抑制フラグ
    """
    # CPU スレッド設定
    logical_cpus = os.cpu_count() or 4
    
    # CPUコア数に応じたスレッド設定
    tf.config.threading.set_intra_op_parallelism_threads(max(1, logical_cpus - 1))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, logical_cpus // 4))
    
    # GPUがあるときの追加設定
    if gpus and not silent:
        print(f"CUDA GPU 加速が有効化されました")


def get_memory_usage() -> Tuple[float, str]:
    """現在のメモリ使用量を取得
    
    Returns:
        (使用メモリ (MB), フォーマット済み文字列)
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        return memory_mb, f"メモリ使用量: {memory_mb:.1f} MB"
    except ImportError:
        return 0.0, "メモリ使用量: 不明 (psutilがインストールされていません)"


class RealtimePlotter:
    """リアルタイムで学習曲線を表示するプロッター"""
    
    def __init__(self, title="Evolution Strategy Learning Curve", figsize=(10, 6)):
        """初期化
        
        Args:
            title: グラフのタイトル
            figsize: 図のサイズ
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # インタラクティブモードを有効化
        plt.ion()
        
        # 図とアックス初期化
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Reward')
        
        # データ保持用
        self.epochs = []
        self.mean_rewards = []
        self.max_rewards = []
        self.min_rewards = []
        
        # プロット用のライン
        self.mean_line, = self.ax.plot([], [], 'b-', label='Mean Reward')
        self.max_line, = self.ax.plot([], [], 'g-', label='Max Reward')
        self.min_line, = self.ax.plot([], [], 'r-', label='Min Reward')
        
        # 凡例表示
        self.ax.legend()
        
        # 初期表示
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update(self, epoch, rewards):
        """プロットを更新
        
        Args:
            epoch: 現在のエポック数
            rewards: 現在のエポックの報酬配列
        """
        import numpy as np
        
        # データ追加
        self.epochs.append(epoch)
        self.mean_rewards.append(np.mean(rewards))
        self.max_rewards.append(np.max(rewards))
        self.min_rewards.append(np.min(rewards))
        
        # ラインデータ更新
        self.mean_line.set_data(self.epochs, self.mean_rewards)
        self.max_line.set_data(self.epochs, self.max_rewards)
        self.min_line.set_data(self.epochs, self.min_rewards)
        
        # 軸の範囲を自動調整
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Y軸の範囲を調整（最小値を0に設定）
        y_min, y_max = self.ax.get_ylim()
        self.ax.set_ylim(0, y_max * 1.1)
        
        # 描画更新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def save(self, filename):
        """現在のプロットを保存
        
        Args:
            filename: 保存先ファイル名
        """
        self.fig.savefig(filename)
        print(f"学習曲線を保存しました: {filename}")