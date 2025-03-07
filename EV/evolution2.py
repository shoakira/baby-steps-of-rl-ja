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
# config.py - 設定と定数
###############################################################################

import os
import platform

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
        """GPUの有無に応じた最適な並列ジョブ数を返す"""
        import tensorflow as tf
        is_using_gpu = any('GPU' in device.name for device in tf.config.list_physical_devices())
        return 4 if is_using_gpu else 7
    
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
    
    if platform.processor() == 'arm':  # Apple Silicon検出
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNNを無効化（MPS互換性向上）
        physical_devices = tf.config.list_physical_devices()
        
        # GPU検出時の出力（silent=Trueなら抑制）
        if any('GPU' in device.name for device in physical_devices) and not silent:
            print("MPS/GPU 加速が有効化されました")
        elif not silent:
            print("CPU モードで実行します")
            
        # マルチスレッド設定（Apple Siliconの8コア活用）
        tf.config.threading.set_intra_op_parallelism_threads(8)  # 演算スレッド数
        tf.config.threading.set_inter_op_parallelism_threads(1)  # 演算グラフ並列数


###############################################################################
# environment.py - 環境ラッパー
###############################################################################

import gymnasium as gym
import numpy as np
from PIL import Image

class CartPoleVectorObserver:
    """CartPole環境の状態ベクトル観測（並列処理に最適化）"""
    def __init__(self, render_mode=None):
        """環境の初期化
        
        Args:
            render_mode (str, optional): 描画モード。デフォルトはNone（描画なし）
        """
        from config import Config
        # render_mode引数で描画モードを制御
        self._env = gym.make(Config.ENV_NAME, render_mode=render_mode)
    
    @property
    def action_space(self):
        """行動空間を返す"""
        return self._env.action_space
    
    @property
    def observation_space(self):
        """観測空間を返す"""
        return self._env.observation_space
        
    def reset(self):
        """環境リセット、状態ベクトルを返す"""
        observation, info = self._env.reset()
        return observation  # [位置, 速度, 角度, 角速度]
        
    def render(self):
        """環境描画（render_mode='human'で初期化されている場合に機能）"""
        self._env.render()
        
    def step(self, action):
        """行動実行、状態ベクトルを返す
        
        Args:
            action: 実行する行動
            
        Returns:
            tuple: (次状態, 報酬, 終了フラグ, 情報)
        """
        n_state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return n_state, reward, done, info


class CartPoleObserver:
    """CartPole環境の画像ベース観測（参照実装、現在は使用していない）"""
    def __init__(self, width, height, frame_count):
        """
        Args:
            width (int): 出力画像幅
            height (int): 出力画像高さ
            frame_count (int): フレーム数
        """
        from config import Config
        self._env = gym.make(Config.ENV_NAME, render_mode="rgb_array")
        self.width = width
        self.height = height

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        """環境リセット、画像観測を返す"""
        observation, info = self._env.reset()
        return self.transform(self._env.render())

    def render(self):
        """環境描画"""
        self._env.render()

    def step(self, action):
        """行動実行、画像観測を返す"""
        n_state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return self.transform(self._env.render()), reward, done, info

    def transform(self, state):
        """RGB画像をグレースケール、リサイズ、正規化
        
        Args:
            state: RGB画像配列
            
        Returns:
            np.array: 変換済み画像
        """
        image = Image.fromarray(state).convert("L")  # グレースケール
        resized = image.resize((self.width, self.height))  # リサイズ
        normalized = np.array(resized) / 255.0  # [0,1]に正規化
        return normalized.reshape((self.height, self.width, 1))  # チャンネル次元追加


###############################################################################
# agent.py - エージェントの実装
###############################################################################

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

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
        
        # 2層ニューラルネット構築
        normal = K.initializers.GlorotNormal()  # Xavier初期化
        inputs = K.Input(shape=(len(state),), dtype=tf.float32)
        x = K.layers.Dense(Config.HIDDEN_LAYER_SIZE, activation="relu", kernel_initializer=normal)(inputs)
        x = K.layers.Dense(Config.HIDDEN_LAYER_SIZE, activation="relu", kernel_initializer=normal)(x)
        outputs = K.layers.Dense(len(self.actions), activation="softmax")(x)
        model = K.Model(inputs=inputs, outputs=outputs)
        self.model = model
        
        # 既存の重みがあれば設定
        if len(weights) > 0:
            self.model.set_weights(weights)
            
        # モデルコンパイル
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # 高速推論関数（tf.function で最適化）
        @tf.function(reduce_retracing=True)  # 再トレース回数削減
        def predict_fn(x):
            return self.model(x, training=False)
            
        self._predict_fn = predict_fn
        
        # ウォームアップ推論（GPU初期化、エラー時はCPU推論へフォールバック）
        try:
            dummy_input = np.zeros((1, len(state)), dtype=np.float32)
            self._predict_fn(tf.convert_to_tensor(dummy_input))
        except Exception as e:
            print(f"警告: GPU推論初期化エラー。CPU推論に切り替えます: {e}")
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


###############################################################################
# trainer.py - 進化戦略トレーナー
###############################################################################

import sys
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class EvolutionalTrainer:
    """進化戦略（ES）によるエージェント訓練"""
    def __init__(self, population_size=20, sigma=0.5, learning_rate=0.1,
                 report_interval=10):
        """
        Args:
            population_size (int): 1世代あたりの個体数
            sigma (float): ノイズの標準偏差
            learning_rate (float): 更新率
            report_interval (int): 報告間隔
        """
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.weights = ()
        self.reward_log = []

    @classmethod
    def experiment(cls, p_index, weights, sigma, episode_per_agent):
        """個体評価実験（並列処理用、シリアライズ可能なクラスメソッド）
        
        Args:
            p_index (int): プロセス番号
            weights (tuple): 重みベクトル
            sigma (float): ノイズの標準偏差
            episode_per_agent (int): 1個体あたりのエピソード数
            
        Returns:
            tuple: (報酬, ノイズ)
        """
        from config import Config
        
        # プロセスごとに独立した乱数シード設定
        seed = int(time.time() * 1000000) % (2**32) + p_index
        np.random.seed(seed)
        
        # メインプロセス（index=0）のみ簡潔な進捗表示
        if p_index == 0:
            sys.stdout.write(f"\r      個体0: エピソード評価中...")
            sys.stdout.flush()
        
        try:
            # エージェント実行で報酬とノイズを取得
            result, debug_info = cls.run_agent(episode_per_agent, weights, sigma, max_step=Config.MAX_STEPS_PER_EPISODE, p_index=p_index)
            
            # メインプロセスの場合、デバッグ情報を表示
            if p_index == 0 and debug_info:
                sys.stdout.write(f"\r      評価完了: {debug_info[:60]}...")
                sys.stdout.flush()
                
            return result
                    
        except Exception as e:
            # エラー時はゼロ報酬を返す
            if p_index == 0:
                sys.stdout.write(f"\r      エラー: {str(e)[:30]}...")
                sys.stdout.flush()
            return (0, [np.zeros_like(w) for w in weights])

    def train(self, epoch=100, episode_per_agent=1, render=False, silent=False):
        """進化戦略によるエージェント訓練（メイン処理）
        
        Args:
            epoch (int): 訓練エポック数
            episode_per_agent (int): 1個体あたりのエピソード数
            render (bool): 描画するかどうか
            silent (bool): 出力を抑制するかどうか
            
        Returns:
            EvolutionalAgent: 訓練されたエージェント
        """
        from environment import CartPoleVectorObserver
        from agent import EvolutionalAgent
        from config import Config
        
        train_start = time.time()
        
        if not silent:
            print("===== 進化戦略学習を開始します =====")
            print(f"エポック数: {epoch}, 集団サイズ: {self.population_size}")
            
        # 並列数決定
        n_jobs = Config.get_parallel_jobs()
        
        if not silent:
            # GPU/CPU検出と表示
            is_using_gpu = any('GPU' in device.name for device in tf.config.list_physical_devices())
            print(f"[1/5] 環境準備: {'GPU計算 + ' if is_using_gpu else ''}CPU並列処理: {n_jobs}個のプロセス")

        # 環境と基本エージェントの準備
        if not silent:
            print("[2/5] モデル初期化中...")
        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()
        
        # 並列処理設定（一度だけ初期化して再利用）
        if not silent:
            print("[3/5] 並列評価エンジン準備中...")
        parallel = Parallel(n_jobs=n_jobs, verbose=0, 
                          prefer="processes",
                          batch_size="auto",
                          backend="multiprocessing",
                          max_nbytes=None)
        
        # エポックごとの学習ループ
        if not silent:
            print(f"[4/5] 学習開始: {epoch}エポック x {self.population_size}個体")
        
        for e in range(epoch):
            epoch_start = time.time()
            
            # 並列個体評価
            if not silent:
                elapsed = time.time() - train_start
                print(f"\r  エポック {e+1}/{epoch}: [タスク 1/2] 個体評価中... (経過時間: {elapsed:.1f}秒)", end="")
                sys.stdout.flush()
                    
            results = parallel(
                delayed(self.__class__.experiment)(
                    i, self.weights, self.sigma, episode_per_agent
                ) for i in range(self.population_size)
            )
            
            # 重み更新
            if not silent:
                elapsed = time.time() - train_start
                print(f"\r  エポック {e+1}/{epoch}: [タスク 2/2] 重み更新中... (経過時間: {elapsed:.1f}秒)", end="")
                sys.stdout.flush()
                    
            self.update(results)
            
            # エポック結果表示
            if not silent:
                rewards = self.reward_log[-1]
                epoch_time = time.time() - epoch_start
                print(f"\r  エポック {e+1}/{epoch}: 報酬 {rewards.mean():.1f} (最大:{rewards.max():.1f})")

        # 学習完了
        if not silent:
            print("[5/5] 学習完了、モデル最終化中...")
            
        # 訓練済みエージェントを返す
        agent.model.set_weights(self.weights)
        return agent

    @classmethod
    def make_env(cls):
        """環境生成ファクトリーメソッド
        
        Returns:
            CartPoleVectorObserver: 生成された環境インスタンス
        """
        from environment import CartPoleVectorObserver
        return CartPoleVectorObserver()

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000, p_index=-1):
        """エージェント実行による評価（個体の適合度計算）
        
        Args:
            episode_per_agent (int): 1個体あたりのエピソード数
            base_weights (tuple): 基本重みベクトル
            sigma (float): ノイズの標準偏差
            max_step (int): 最大ステップ数
            p_index (int): プロセス番号
            
        Returns:
            tuple: ((報酬, ノイズ), デバッグ情報)
        """
        from agent import EvolutionalAgent
        import time
        
        debug_info = []  # デバッグ情報
        detailed_logs = []  # 詳細ログ
        start_time = time.time()
        
        try:
            # 各実行で新しい環境作成（並列処理でのリソース競合防止）
            env_start = time.time()
            env = cls.make_env()
            actions = list(range(env.action_space.n))
            env_time = time.time() - env_start
            debug_info.append(f"環境:{env_time:.1f}秒")
            
            # デバッグ表示（メインプロセスのみ）
            if p_index == 0:
                sys.stdout.write(f"\r    [1/4] 環境構築: {env_time:.1f}秒")
                sys.stdout.flush()

            # エージェント作成
            agent_start = time.time()
            agent = EvolutionalAgent(actions)
            agent_time = time.time() - agent_start
            debug_info.append(f"エージェント:{agent_time:.1f}秒")
            
            if p_index == 0:
                sys.stdout.write(f"\r    [2/4] エージェント初期化: {agent_time:.1f}秒")
                sys.stdout.flush()

            # ノイズ付き重みベクトル生成
            noise_start = time.time()
            noises = []  # 適用したノイズを記録
            new_weights = []  # ノイズ適用済み重み
            for w in base_weights:
                noise = np.random.randn(*w.shape)  # ガウス分布ノイズ
                new_weights.append(w + sigma * noise)  # ノイズ適用
                noises.append(noise)  # ノイズ記録（勾配計算用）
            noise_time = time.time() - noise_start
            debug_info.append(f"ノイズ:{noise_time:.1f}秒")
            
            if p_index == 0:
                sys.stdout.write(f"\r    [3/4] ノイズ生成: {noise_time:.1f}秒")
                sys.stdout.flush()

            # エージェント評価
            total_reward = 0
            episode_count = 0  # 完了エピソード数
            eval_start = time.time()
            
            for e in range(episode_per_agent):
                try:
                    # メインプロセス (p_index=0) の場合のみ進捗表示
                    if p_index == 0:
                        sys.stdout.write(f"\r    [4/4] エピソード評価: {e+1}/{episode_per_agent} " + 
                                       f"(完了:{episode_count})")
                        sys.stdout.flush()
                        
                    episode_start = time.time()
                    s = env.reset()
                    if s is None:
                        continue  # リセット失敗時はスキップ
                        
                    # エージェント初期化（初回のみ）
                    if agent.model is None:
                        agent.initialize(s, new_weights)
                    
                    done = False
                    step = 0
                    episode_reward = 0
                    
                    # エピソード実行
                    while not done and step < max_step:
                        # 10ステップごとに進捗更新（メインプロセスのみ）
                        if p_index == 0 and step % 10 == 0:
                            sys.stdout.write(f"\r    [4/4] エピソード評価: {e+1}/{episode_per_agent} " + 
                                           f"(完了:{episode_count}, 実行中:ステップ{step}, 報酬:{episode_reward:.1f})")
                            sys.stdout.flush()
                        
                        # 行動選択と環境ステップ
                        a = agent.policy(s)
                        n_state, reward, done, info = env.step(a)
                        episode_reward += reward
                        s = n_state
                        step += 1

                    # エピソード完了
                    episode_time = time.time() - episode_start
                    detailed_logs.append(f"エピソード{e+1}: {step}ステップ, 報酬{episode_reward:.1f} ({episode_time:.1f}秒)")
                    total_reward += episode_reward
                    episode_count += 1
                    
                except Exception as e:
                    detailed_logs.append(f"エピソード{e+1}エラー: {str(e)}")
                    continue  # エラー時は次のエピソードへ

            # 評価終了
            eval_time = time.time() - eval_start
            debug_info.append(f"評価:{eval_time:.1f}秒")
            
            # 詳細ログをまとめる
            debug_summary = ", ".join(debug_info)
            detail = " | ".join(detailed_logs)
            
            # 平均報酬計算（完了エピソードのみ）
            reward = total_reward / max(episode_count, 1)
            return (reward, noises), f"{debug_summary} | {detail}"  # 報酬とノイズとデバッグ情報を返す
            
        except Exception as e:
            # 全体エラー時はゼロ報酬を返す
            return (0, [np.zeros_like(w) for w in base_weights]), f"エラー: {str(e)}"

    def update(self, agent_results):
        """収集した結果から重みを更新（進化戦略の核心部分）
        
        Args:
            agent_results (list): 各エージェントの評価結果リスト
        """
        # 有効な結果のみフィルタリング
        valid_results = [r for r in agent_results if isinstance(r, tuple) and len(r) == 2 and r[0] is not None and r[1] is not None]
        
        if not valid_results:
            print("WARNING: No valid agent results to update weights")
            return
            
        # 報酬抽出
        rewards = np.array([r[0] for r in valid_results])
        
        # ノイズ抽出（型チェック）
        noises_list = []
        for r in valid_results:
            if isinstance(r[1], list) and len(r[1]) == len(self.weights):
                noises_list.append(r[1])
        
        if not noises_list:
            print("WARNING: No valid noise data found")
            return
            
        # 報酬の正規化（平均0、標準偏差1）
        if len(rewards) > 1:
            normalized_rs = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        else:
            normalized_rs = np.zeros_like(rewards)

        # 重み更新
        new_weights = []
        for i, w in enumerate(self.weights):
            try:
                # i番目のレイヤーのノイズを抽出
                noise_at_i = [n[i] for n in noises_list]
                
                # 各ノイズ × 正規化報酬 の積和計算
                update_val = np.zeros_like(w)
                for ni, r in zip(noise_at_i, normalized_rs):
                    try:
                        update_val += ni * r  # 報酬加重ノイズの合計
                    except Exception as e:
                        print(f"Error adding noise contribution: {e}")
                
                # 勾配的な更新式
                rate = self.learning_rate / ((len(noises_list) + 1e-10) * self.sigma)
                w = w + rate * update_val  # 重み更新
            except Exception as e:
                print(f"Weight update error at index {i}: {e}")
            new_weights.append(w)

        # 更新された重みを保存
        self.weights = new_weights
        self.reward_log.append(rewards)  # 報酬履歴に追加

    def log(self):
        """エポック結果の表示"""
        rewards = self.reward_log[-1]
        print("Epoch {}: reward {:.3f}(max:{:.1f}, min:{:.1f})".format(
            len(self.reward_log), rewards.mean(),
            rewards.max(), rewards.min()))

    def plot_rewards(self, save_path=None, y_max=None):
        """学習曲線のプロット（ファイル保存機能追加、縦軸範囲指定可能）
        
        Args:
            save_path (str, optional): 保存ファイル名（省略時は自動生成）
            y_max (float, optional): 縦軸の最大値（省略時はデータに基づき自動設定）
        """
        from config import Config
        
        # プロットデータ準備
        indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        
        # グラフサイズ設定
        plt.figure(figsize=(10, 6))
        plt.title(f"Evolution Strategy Learning Curve (pop={self.population_size}, σ={self.sigma}, lr={self.learning_rate})")
        plt.grid()
        
        # データプロット
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")  # 標準偏差範囲
        plt.plot(indices, means, "o-", color="g",
                 label=f"Reward (final: {means[-1]:.1f})")  # 平均報酬
        
        # ラベル設定
        plt.xlabel("Epochs")
        plt.ylabel("Average Reward")
        plt.legend(loc="best")
        
        # 縦軸の範囲設定
        max_reward = means.max() + stds.max()  # データの最大値+標準偏差
        
        if y_max is None:
            # 自動設定: データの最大値+余白、または500のうち大きい方
            if max_reward < 450:  # データが十分小さければ自動調整
                plt.ylim(0, max_reward * 1.1)  # 10%のマージン
            else:
                plt.ylim(0, 550)  # CartPoleの最大値+マージン
        else:
            # 指定値を使用
            plt.ylim(0, y_max)
        
        # 500報酬の線を追加（CartPoleのゴール）
        if max_reward > 100 or y_max is None or y_max >= 500:  # 報酬が十分大きい場合のみ表示
            plt.axhline(y=500, color='r', linestyle='--', alpha=0.3, label="Solved")
        
        # plotfileフォルダの保存処理
        plot_dir = Config.get_plot_dir()
        
        # 保存パス設定
        if save_path:
            # 指定されたファイル名をplotfileフォルダ内に配置
            filename = os.path.basename(save_path)
            final_path = os.path.join(plot_dir, filename)
        else:
            # 自動ファイル名生成
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"es_results_{timestamp}.png"
            final_path = os.path.join(plot_dir, filename)
        
        # ファイル保存
        plt.savefig(final_path, dpi=100, bbox_inches='tight')
        print(f"学習曲線を保存しました: {final_path}")
        
        # 画面表示も行う
        plt.show()


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