"""
# 進化戦略（Evolution Strategy）強化学習アルゴリズム

## 概要
パラメータ空間での勾配を使わない最適化手法。各世代でパラメータにランダムノイズを加え、
報酬に基づいて良い方向に更新する。

## 入出力
入力: CartPole環境からの状態ベクトル [位置、速度、角度、角速度]
出力: 行動確率（2つの行動 - カートを左右に動かす）

## アルゴリズム
1. 基本パラメータにノイズを加えた複数候補を生成
2. 各候補をエージェントで評価し報酬を計算
3. 報酬が高かった方向に重みを更新
4. 繰り返して最適方策を学習

## コマンド書式
python evolution.py [オプション]

オプション:
  --play            学習済みモデルでプレイ
  --epochs N        学習エポック数（デフォルト: 100）
  --pop-size N      集団サイズ（デフォルト: 100）
  --sigma X         探索ノイズ幅（デフォルト: 0.1）
  --lr X            学習率（デフォルト: 0.05）
  --silent          警告・情報メッセージを抑制

例:
  python evolution.py --epochs 200 --pop-size 120 --sigma 0.1

実装: TensorFlow + Gymnasium + joblib（並列処理）
環境: CartPole-v1
"""

# ===== ライブラリのインポート =====
import os, sys, time
import argparse  # コマンドライン引数処理
import numpy as np  # 数値計算
import datetime  # 時刻と日付の処理
from joblib import Parallel, delayed  # 並列処理
from PIL import Image  # 画像処理
import matplotlib.pyplot as plt  # グラフ描画
import gymnasium as gym  # 強化学習環境
import platform  # プラットフォーム情報

# ===== Apple Silicon最適化 =====
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # M1/M2/M3チップの8コア活用
np.random.seed(0)  # 再現性確保

# ===== TensorFlow環境設定 =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TFの警告抑制
import tensorflow as tf
import tensorflow.keras as K

# GPU/MPS対応TensorFlow設定関数
def configure_tensorflow(silent=False):
    """Apple SiliconのGPU/MPS加速を設定"""
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


# ===== 強化学習エージェントの実装 =====
class EvolutionalAgent():
    """進化戦略で学習するエージェント"""
    def __init__(self, actions):
        self.actions = actions  # 行動空間
        self.model = None  # NNモデル
        self._predict_fn = None  # 高速推論用キャッシュ
        
    def save(self, model_path):
        """モデル保存（.keras形式）"""
        if model_path.endswith('.h5'):
            model_path = model_path.replace('.h5', '.keras')
        # save_formatを削除するだけで解決
        self.model.save(model_path, overwrite=True)

    @classmethod
    def load(cls, env, model_path):
        """保存済みモデル読み込み"""
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
        """ニューラルネットワークモデルの初期化"""
        # 2層ニューラルネット構築
        normal = K.initializers.GlorotNormal()  # Xavier初期化
        inputs = K.Input(shape=(len(state),), dtype=tf.float32)
        x = K.layers.Dense(24, activation="relu", kernel_initializer=normal)(inputs)
        x = K.layers.Dense(24, activation="relu", kernel_initializer=normal)(x)
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
        """状態から確率的に行動を選択"""
        # 状態をTensorFlow形式に変換
        state_tensor = tf.convert_to_tensor(
            np.array([state], dtype=np.float32)
        )
        # モデルで行動確率を計算し、確率的にサンプリング
        action_probs = self._predict_fn(state_tensor)[0].numpy()
        action = np.random.choice(self.actions, size=1, p=action_probs)[0]
        return action

    def play(self, env, episode_count=5, render=True):
        """学習済みエージェントで環境を実行"""
        for e in range(episode_count):
            s = env.reset()  # 環境リセット
            done = False
            terminated = False
            truncated = False
            episode_reward = 0
            
            # エピソード実行
            while not done:
                if render:
                    env.render()  # 可視化
                a = self.policy(s)  # 行動選択
                n_state, reward, done, info = env.step(a)  # ここは4つの値を返す
                episode_reward += reward
                s = n_state
            
            print(f"エピソード {e+1}: 報酬 {episode_reward}")


# ===== 環境ラッパー（画像ベース、未使用） =====
class CartPoleObserver():
    """CartPole環境の画像ベース観測（現在は使用していない）"""
    def __init__(self, width, height, frame_count):
        self._env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.width = width    # 出力画像幅
        self.height = height  # 出力画像高さ

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
        """RGB画像をグレースケール、リサイズ、正規化"""
        image = Image.fromarray(state).convert("L")  # グレースケール
        resized = image.resize((self.width, self.height))  # リサイズ
        normalized = np.array(resized) / 255.0  # [0,1]に正規化
        return normalized.reshape((self.height, self.width, 1))  # チャンネル次元追加


# ===== 環境ラッパー（ベクトルベース、使用中） =====
class CartPoleVectorObserver():
    """CartPole環境の状態ベクトル観測（並列処理に最適化）"""
    def __init__(self):
        # render_mode=None で描画なし（並列処理時の効率化）
        self._env = gym.make("CartPole-v1", render_mode=None)
    
    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.observation_space
        
    def reset(self):
        """環境リセット、状態ベクトルを返す"""
        observation, info = self._env.reset()
        return observation  # [位置, 速度, 角度, 角速度]
        
    def render(self):
        """環境描画（並列処理では未使用）"""
        self._env.render()
        
    def step(self, action):
        """行動実行、状態ベクトルを返す"""
        n_state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return n_state, reward, done, info


# ===== 進化戦略トレーナー =====
class EvolutionalTrainer():
    """進化戦略（ES）によるエージェント訓練"""
    def __init__(self, population_size=20, sigma=0.5, learning_rate=0.1,
                 report_interval=10):
        self.population_size = population_size  # 1世代あたりの個体数
        self.sigma = sigma                      # ノイズの標準偏差
        self.learning_rate = learning_rate      # 更新率
        self.weights = ()                       # 現在のニューラルネット重み
        self.reward_log = []                    # 報酬履歴

    @classmethod
    def experiment(cls, p_index, weights, sigma, episode_per_agent):
        """個体評価実験（並列処理用、シリアライズ可能なクラスメソッド）"""
        # プロセスごとに独立した乱数シード設定
        seed = int(time.time() * 1000000) % (2**32) + p_index
        np.random.seed(seed)
        
        # メインプロセス（index=0）のみ簡潔な進捗表示
        if p_index == 0:
            sys.stdout.write(f"\r      個体0: エピソード評価中...")
            sys.stdout.flush()
        
        try:
            # エージェント実行で報酬とノイズを取得
            result, debug_info = cls.run_agent(episode_per_agent, weights, sigma)
            
            # メインプロセスの場合、デバッグ情報を表示
            if p_index == 0 and debug_info:
                sys.stdout.write(f"\r      評価完了: {debug_info[:60]}...")
                sys.stdout.flush()
                
            # 重要: 第1要素のみを返す（(reward, noises)のタプル）
            return result  # ここを修正
                    
        except Exception as e:
            # エラー時はゼロ報酬を返す
            if p_index == 0:
                sys.stdout.write(f"\r      エラー: {str(e)[:30]}...")
                sys.stdout.flush()
            return (0, [np.zeros_like(w) for w in weights])

    def train(self, epoch=100, episode_per_agent=1, render=False, silent=False):
        """進化戦略によるエージェント訓練（メイン処理）"""
        train_start = time.time()
        
        if not silent:
            print("===== 進化戦略学習を開始します =====")
            print(f"エポック数: {epoch}, 集団サイズ: {self.population_size}")
            
        # GPU/CPU検出と並列数決定
        is_using_gpu = any('GPU' in device.name for device in tf.config.list_physical_devices())
        n_jobs = 4 if is_using_gpu else 7
        
        if not silent:
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
            
            # エポック結果表示（これは残す）
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
        """環境生成ファクトリーメソッド"""
        return CartPoleVectorObserver()

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000, p_index=-1):
        """エージェント実行による評価（個体の適合度計算）"""
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
        """収集した結果から重みを更新（進化戦略の核心部分）"""
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
            save_path: 保存ファイル名（省略時は自動生成）
            y_max: 縦軸の最大値（省略時はデータに基づき自動設定）
        """
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
        
        # plotfileフォルダの作成と保存処理
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(base_dir, "plotfile")
        
        # フォルダが存在しなければ作成
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
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


# ===== メイン処理 =====
def main(play, epochs, pop_size, sigma, lr, silent=False):
    """メインエントリポイント関数（学習または実行）"""
    start_time = datetime.datetime.now()
    
    # サイレントモード設定
    if silent:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import warnings
        warnings.filterwarnings('ignore')
    else:
        print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # モデル保存パス設定
    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.keras")
    
    if play:
        # render_mode="human"を指定して可視化環境を作成
        env = gym.make("CartPole-v1", render_mode="human")
        env = CartPoleVectorObserver()  
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
            episode_per_agent=5,      # 1個体あたりのエピソード数
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


# ===== スクリプト実行時のエントリーポイント =====
if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true", help="学習済みモデルで実行")
    parser.add_argument("--epochs", type=int, default=100, help="学習エポック数")
    parser.add_argument("--pop-size", type=int, default=100, help="集団サイズ")
    parser.add_argument("--sigma", type=float, default=0.1, help="探索のノイズ幅")
    parser.add_argument("--lr", type=float, default=0.05, help="学習率")
    parser.add_argument("--silent", action="store_true", help="警告を抑制")

    # 引数解析
    args = parser.parse_args()
    
    # 警告抑制設定（静かモードの場合）
    if args.silent:
        import warnings
        warnings.filterwarnings('ignore')
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # TensorFlow設定（GPUの初期化など）
    configure_tensorflow(silent=args.silent)
    
    # メイン処理実行
    main(args.play, args.epochs, args.pop_size, args.sigma, args.lr, args.silent)
