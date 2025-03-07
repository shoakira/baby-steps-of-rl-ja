###############################################################################
# config.py - 設定と定数
###############################################################################

import os
import platform
import tensorflow as tf

class Config:
    """進化戦略およびアプリケーション全体の設定"""
    
    # TensorFlow/GPU設定
    TF_LOG_LEVEL = "3"  # TFの警告抑制レベル
    MAX_THREADS = "8"   # Apple Siliconの8コア活用
    SEED = 0           # 再現性確保のためのシード値
    
    # 学習パラメータのデフォルト値
    DEFAULT_EPOCHS = 100
    DEFAULT_POPULATION_SIZE = 100
    DEFAULT_SIGMA = 0.1
    DEFAULT_LEARNING_RATE = 0.05
    DEFAULT_EPISODE_PER_AGENT = 5
    
    # 環境設定
    ENV_NAME = "CartPole-v1"
    MAX_STEPS_PER_EPISODE = 1000
    
    # モデル設定
    HIDDEN_LAYER_SIZE = 24
    MODEL_FILENAME = "ev_agent.keras"
    
    # 並列処理設定
    @staticmethod
    def get_parallel_jobs():
        """GPUの有無に応じた最適な並列ジョブ数を返す
        
        より少ない並列ジョブ数に設定し、リソース競合を防止します。
        本番環境では状況に応じて調整してください。
        """
        import tensorflow as tf
        is_using_gpu = any('GPU' in device.name for device in tf.config.list_physical_devices())
        # 並列ジョブ数を少なめに設定して、リソース競合を減らす
        return 2 if is_using_gpu else 4
    
    # ファイルパス
    @staticmethod
    def get_model_path():
        """モデル保存先のフルパスを返す"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, Config.MODEL_FILENAME)
    
    @staticmethod
    def get_plot_dir():
        """プロット保存先ディレクトリを返す"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(base_dir, "plotfile")
        # フォルダがなければ作成
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        return plot_dir
