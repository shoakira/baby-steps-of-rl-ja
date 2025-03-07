###############################################################################
# agent.py - エージェントの実装
###############################################################################

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import sys

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
        # 拡張子の自動検出
        if model_path.endswith('.h5') and not os.path.exists(model_path):
            keras_path = model_path.replace('.h5', '.keras')
            if os.path.exists(keras_path):
                model_path = keras_path
        
        # エージェント作成とモデル読み込み
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        
        # 高速推論関数の設定（読み込み時に初期化）
        @tf.function(reduce_retracing=True)
        def predict_fn(x):
            return agent.model(x, training=False)
        
        agent._predict_fn = predict_fn
        
        # ウォームアップ推論
        dummy_input = np.zeros((1, env.observation_space.shape[0]), dtype=np.float32)
        try:
            agent._predict_fn(tf.convert_to_tensor(dummy_input))
        except Exception as e:
            print(f"警告: GPU推論初期化エラー。CPU推論に切り替えます: {e}")
            agent._predict_fn = lambda x: agent.model.predict(x, verbose=0)
        
        return agent

    def initialize(self, state, weights=()):
        """ニューラルネットワークモデルの初期化
        
        Args:
            state: 初期状態
            weights (tuple, optional): 初期重み
        """
        from config import Config
        import time
        
        # 2層ニューラルネット構築
        normal = K.initializers.GlorotNormal()  # Xavier初期化
        inputs = K.Input(shape=(len(state),), dtype=tf.float32)
        x = K.layers.Dense(Config.HIDDEN_LAYER_SIZE, activation="relu", kernel_initializer=normal)(inputs)
        x = K.layers.Dense(Config.HIDDEN_LAYER_SIZE, activation="relu", kernel_initializer=normal)(x)
        outputs = K.layers.Dense(len(self.actions), activation="softmax")(x)
        
        # 混合精度を有効化（FP16とFP32）- パフォーマンス向上
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        model = K.Model(inputs=inputs, outputs=outputs)
        self.model = model
        
        # モデルサマリー出力（デバッグ用）
        # print(f"モデルサマリー:\n{model.summary()}")
        
        # 既存の重みがあれば設定
        if len(weights) > 0:
            self.model.set_weights(weights)
            
        # モデルコンパイル - 最適化設定
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        
        # 高速推論関数（tf.function で最適化）
        @tf.function(reduce_retracing=True)
        def predict_fn(x):
            return self.model(x, training=False)
            
        self._predict_fn = predict_fn
        
        # ウォームアップ推論（GPU初期化、エラー時はCPU推論へフォールバック）
        try:
            # テンポラル表示に変更
            sys.stdout.write("\rモデル初期化中: ウォームアップ推論を実行...")
            sys.stdout.flush()
            start_time = time.time()
            
            # 複数回のウォームアップ実行で初期化を確実に
            dummy_input = np.zeros((1, len(state)), dtype=np.float32)
            tensor_input = tf.convert_to_tensor(dummy_input)
            
            # 3回繰り返してコンパイルを確実に
            for i in range(3):
                _ = self._predict_fn(tensor_input)
                # 進捗をテンポラル表示
                sys.stdout.write(f"\rモデル初期化中: ウォームアップ推論 {i+1}/3 実行中...")
                sys.stdout.flush()
            
            # 完了表示（最終行はクリアしない）
            elapsed = time.time() - start_time
            # メインスレッド/プロセスのみ表示（オプション）
            if hasattr(self, 'is_main_process') and self.is_main_process:
                print(f"\rモデル初期化完了: {elapsed:.3f}秒           ")
            else:
                # 何も表示しない（必要に応じてコメントアウト）
                pass
                
        except Exception as e:
            print(f"警告: GPU推論初期化エラー。CPU推論に切り替えます: {e}")
            # CPU推論ではverbose=0でメッセージを抑制
            self._predict_fn = lambda x: self.model.predict(x, verbose=0)

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
