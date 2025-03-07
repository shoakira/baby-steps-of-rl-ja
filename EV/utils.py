###############################################################################
# utils.py - ユーティリティ関数
###############################################################################

import os
import tensorflow as tf
import platform

def configure_tensorflow(silent=False):
    """Apple SiliconのGPU/MPS加速を含むTensorFlow環境設定
    
    Args:
        silent (bool): 警告・情報メッセージを抑制するかどうか
    """
    from config import Config
    
    # 環境変数設定
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = Config.TF_LOG_LEVEL
    os.environ["VECLIB_MAXIMUM_THREADS"] = Config.MAX_THREADS
    
    # TensorFlowのメモリ成長を有効化（メモリ使用量を抑制）
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            if not silent:
                print(f"デバイス {device.name} のメモリ動的割り当てを有効化しました")
        except:
            # 既に初期化されているか、GPUが存在しない場合
            pass
    
    if platform.processor() == 'arm':  # Apple Silicon検出
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNNを無効化（MPS互換性向上）
        physical_devices = tf.config.list_physical_devices()
        
        # GPU検出時の出力（silent=Trueなら抑制）
        if any('GPU' in device.name for device in physical_devices) and not silent:
            print("MPS/GPU 加速が有効化されました")
        elif not silent:
            print("CPU モードで実行します")
            
        # マルチスレッド設定（Apple Siliconの8コア活用）
        tf.config.threading.set_intra_op_parallelism_threads(6)  # 演算スレッド数を少し減らす
        tf.config.threading.set_inter_op_parallelism_threads(1)  # 演算グラフ並列数
    
    # グローバルでメモリ使用量を制限
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and not silent:
        print(f"検出されたGPU: {len(gpus)}個")
        # GPUメモリの制限（パフォーマンスに影響するため慎重に設定）
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
            if not silent:
                print("GPU メモリ使用量を制限しました (1GB/GPU)")
        except RuntimeError as e:
            if not silent:
                print(f"GPU設定エラー: {e}")