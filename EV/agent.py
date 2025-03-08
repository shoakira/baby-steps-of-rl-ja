###############################################################################
# agent.py - エージェントの実装
###############################################################################

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import sys
import time

class EvolutionalAgent:
    """進化戦略で学習するエージェント"""
    def __init__(self, actions):
        """
        Args:
            actions (list): 利用可能な行動のリスト
        """
        self.actions = actions  # 行動空間
        self.model = None      # NNモデル
        self._predict_fn = None  # 高速推論用キャッシュ
        self.is_main_process = True  # デフォルトではメインプロセスとみなす
        
    def save(self, model_path):
        """モデル保存（.keras形式）
        
        Args:
            model_path (str): 保存先パス
        """
        if model_path.endswith('.h5'):
            model_path = model_path.replace('.h5', '.keras')
        self.model.save(model_path, overwrite=True)

    @classmethod
    def load(cls, env, model_path):
        """保存済みモデル読み込み
        
        Args:
            env: 環境
            model_path (str): モデルファイルパス
            
        Returns:
            EvolutionalAgent: 読み込まれたエージェント
        """
        from config import Config
        
        # 拡張子の自動検出
        if model_path.endswith('.h5') and not os.path.exists(model_path):
            keras_path = model_path.replace('.h5', '.keras')
            if os.path.exists(keras_path):
                model_path = keras_path
        
        # エージェント作成とモデル読み込み
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        
        # 推論関数の設定とウォームアップ
        state_shape = env.observation_space.shape[0]
        dummy_state = np.zeros((1, state_shape), dtype=np.float32)
        agent._setup_predict_fn(dummy_state, is_load=True)
        
        return agent

    def _setup_predict_fn(self, state, is_load=False):
        """推論関数の設定とウォームアップ"""
        from config import Config
        
        # JIT最適化を無効化し、シンプルな関数呼び出しに置き換え
        # 並列処理時のPickle化問題を回避し、安定性を向上
        self._predict_fn = lambda x: self.model(x, training=False)
        
        # ウォームアップ推論（シンプル版）
        try:
            # テンポラル表示 - 並列処理時に出力が混乱するので条件を厳格化
            show_logs = not is_load and hasattr(self, 'is_main_process') and self.is_main_process
            start_time = time.time()
            
            # ウォームアップ用入力の準備
            dummy_input = np.zeros((1, len(state)), dtype=np.float32)
            tensor_input = tf.convert_to_tensor(dummy_input)
            
            # 1回だけ実行（グラフコンパイルなし）
            _ = self._predict_fn(tensor_input)
            
            # 完了表示
            elapsed = time.time() - start_time
            if show_logs:
                print(f"\rモデル初期化完了: {elapsed:.3f}秒" + " " * 20)
                
        except Exception as e:
            # エラー表示もメインプロセスのみ
            if not is_load and hasattr(self, 'is_main_process') and self.is_main_process:
                print(f"警告: 推論初期化エラー。標準推論に切り替えます: {e}")
            # エラー時は標準推論（verbose=0でメッセージを抑制）
            self._predict_fn = lambda x: self.model.predict(x, verbose=0)

    def initialize(self, state, weights=()):
        """ニューラルネットワークモデルの初期化
        
        Args:
            state: 初期状態
            weights (tuple, optional): 初期重み
        """
        from config import Config
        import time
        
        # 並列処理時はCPUモードを強制（GPU競合回避）
        if not hasattr(self, 'is_main_process') or not self.is_main_process:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            os.environ['TF_DISABLE_GPU'] = '1'
            os.environ['DISABLE_MPS'] = '1'  # Apple Silicon MPS も無効化
        
        # 2層ニューラルネット構築
        normal = K.initializers.GlorotNormal()  # Xavier初期化
        inputs = K.Input(shape=(len(state),), dtype=tf.float32)
        x = K.layers.Dense(Config.HIDDEN_LAYER_SIZE, activation="relu", kernel_initializer=normal)(inputs)
        x = K.layers.Dense(Config.HIDDEN_LAYER_SIZE, activation="relu", kernel_initializer=normal)(x)
        outputs = K.layers.Dense(len(self.actions), activation="softmax")(x)
        
        # 混合精度設定
        # float32はApple SiliconのMPS互換性のため使用（float16より安定）
        # 本番環境ではConfig.USE_MIXED_PRECISIONで制御
        if Config.USE_MIXED_PRECISION:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
        else:
            policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # モデル作成
        model = K.Model(inputs=inputs, outputs=outputs)
        self.model = model
        
        # モデルサマリー出力（デバッグモード時のみ）
        if Config.DEBUG_MODEL_SUMMARY:
            print(f"モデルサマリー:")
            model.summary()
        
        # 既存の重みがあれば設定
        if len(weights) > 0:
            self.model.set_weights(weights)
            
        # モデルコンパイル - 最適化設定
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        
        # 推論関数の設定とウォームアップ
        self._setup_predict_fn(state)

    def policy(self, state):
        """状態から確率的に行動を選択
        
        Args:
            state: 現在の状態
            
        Returns:
            int: 選択された行動
        """
        # 状態をTensorFlow形式に変換
        state_tensor = tf.convert_to_tensor(
            np.array([state], dtype=np.float32)
        )
        # モデルで行動確率を計算し、確率的にサンプリング
        action_probs = self._predict_fn(state_tensor)[0].numpy()
        action = np.random.choice(self.actions, size=1, p=action_probs)[0]
        return action

    def play(self, env, episode_count=5, render=True):
        """学習済みエージェントで環境を実行
        
        Args:
            env: 環境
            episode_count (int): 実行するエピソード数
            render (bool): 描画を行うかどうか
        """
        from config import Config
        
        for e in range(episode_count):
            s = env.reset()  # 環境リセット
            done = False
            episode_reward = 0
            
            # エピソード実行
            while not done:
                if render:
                    env.render()  # 可視化
                a = self.policy(s)  # 行動選択
                n_state, reward, done, info = env.step(a)  # 環境更新
                episode_reward += reward
                s = n_state
            
            print(f"エピソード {e+1}: 報酬 {episode_reward}")
