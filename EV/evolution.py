"""
進化戦略（Evolution Strategy）による強化学習

このコードは進化戦略を用いた強化学習エージェントの実装です。
進化戦略とは、勾配計算を使わずに最適なパラメータを探索する手法です。

【アルゴリズムの概要】
1. 基本の重み（パラメータ）に対してランダムなノイズを加えた複数の候補を生成
2. 各候補でエージェントを評価し、環境との相互作用から報酬を計算
3. 報酬が高かったノイズの方向に向けて重みを更新
4. これを繰り返し、最適な方策を学習

【特徴】
- 並列計算が容易（各候補は独立して評価可能）
- 実装がシンプル
- 局所解に陥りにくい
- 勾配が計算できない問題にも適用可能

【使用方法】
- 学習実行: python evolution.py
- 学習済みモデルでプレイ: python evolution.py --play

実装: Gymnasium + TensorFlow + joblib
環境: CartPole-v1 (デフォルト)
"""

import os, sys
import argparse
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
import matplotlib.pyplot as plt
import gymnasium as gym  # gymからgymnasiumに変更
import platform

# ファイル冒頭に追加
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Apple Silicon用に最適化
np.random.seed(0)  # 一貫性のある結果を得るために設定

# Disable TensorFlow GPU for parallel execution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.keras as K

# TensorFlow設定を最適化（ファイル冒頭に一度だけ実行）
def configure_tensorflow(silent=False):
    # Apple Siliconの場合のみMPSを設定
    if platform.processor() == 'arm':
        # TensorFlow-Metal向け環境変数
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # TF設定
        physical_devices = tf.config.list_physical_devices()
        if any('GPU' in device.name for device in physical_devices) and not silent:
            print("ゲホッゲホMPS/GPU 加速が有効化されました")
        elif not silent:
            print("CPU モードで実行します")

        # スレッド最適化
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # ログレベルを最も厳格に設定
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class EvolutionalAgent():
    def __init__(self, actions):
        self.actions = actions
        self.model = None
        self._predict_fn = None  # 推論関数をキャッシュ
        
    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        return agent

    def initialize(self, state, weights=()):
        # 以前と同じモデル構築
        normal = K.initializers.GlorotNormal()
        inputs = K.Input(shape=(len(state),), dtype=tf.float32)
        x = K.layers.Dense(24, activation="relu", kernel_initializer=normal)(inputs)
        x = K.layers.Dense(24, activation="relu", kernel_initializer=normal)(x)
        outputs = K.layers.Dense(len(self.actions), activation="softmax")(x)
        model = K.Model(inputs=inputs, outputs=outputs)
        self.model = model
        
        if len(weights) > 0:
            self.model.set_weights(weights)
            
        # モデルをコンパイル
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # JITコンパイルを使わない安全な関数定義
        @tf.function(reduce_retracing=True)  # jit_compile=True を削除
        def predict_fn(x):
            return self.model(x, training=False)
            
        self._predict_fn = predict_fn
        
        # ウォームアップ推論（try-except で囲む）
        try:
            dummy_input = np.zeros((1, len(state)), dtype=np.float32)
            self._predict_fn(tf.convert_to_tensor(dummy_input))
        except Exception as e:
            print(f"警告: GPU推論初期化エラー。CPU推論に切り替えます: {e}")
            # エラー時はモデル直接呼び出しにフォールバック
            self._predict_fn = lambda x: self.model.predict(x, verbose=0)

    def policy(self, state):
        # 一貫した型とバッチサイズでテンソル化
        state_tensor = tf.convert_to_tensor(
            np.array([state], dtype=np.float32)
        )
        # キャッシュした関数で高速推論
        action_probs = self._predict_fn(state_tensor)[0].numpy()
        action = np.random.choice(self.actions, size=1, p=action_probs)[0]
        return action

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s, _ = env.reset()  # 新しいAPI
            done = False
            terminated = False
            truncated = False
            episode_reward = 0
            
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, terminated, truncated, info = env.step(a)  # 新しいAPI
                done = terminated or truncated
                episode_reward += reward
                s = n_state
            
            print(f"エピソード {e+1}: 報酬 {episode_reward}")


# 代替環境：CartPole-v1はGymnasiumで動作する標準環境
class CartPoleObserver():

    def __init__(self, width, height, frame_count):
        self._env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.width = width
        self.height = height

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        observation, info = self._env.reset()
        return self.transform(self._env.render())

    def render(self):
        self._env.render()

    def step(self, action):
        n_state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return self.transform(self._env.render()), reward, done, info

    def transform(self, state):
        # 画像を取得してグレースケール化
        image = Image.fromarray(state).convert("L")
        # リサイズ
        resized = image.resize((self.width, self.height))
        # 正規化
        normalized = np.array(resized) / 255.0
        # 次元追加（チャネル）
        return normalized.reshape((self.height, self.width, 1))


# CartPoleObserverの代わりにこちらを使用
class CartPoleVectorObserver():
    def __init__(self):
        # レンダリングなしでメモリ使用量を削減
        self._env = gym.make("CartPole-v1", render_mode=None)
    
    @property
    def action_space(self):
        return self._env.action_space
    
    @property
    def observation_space(self):
        return self._env.observation_space
        
    def reset(self):
        observation, info = self._env.reset()
        return observation  # 直接状態ベクトルを返す
        
    def render(self):
        self._env.render()
        
    def step(self, action):
        n_state, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return n_state, reward, done, info  # 直接状態ベクトルを返す


class EvolutionalTrainer():

    def __init__(self, population_size=20, sigma=0.5, learning_rate=0.1,
                 report_interval=10):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = ()
        self.reward_log = []

    @classmethod
    def experiment(cls, p_index, weights, sigma, episode_per_agent):
        """シリアライズ可能なクラスメソッドとして実験関数を定義"""
        import time
        import numpy as np
        import os
        
        # すべてのプロセスで警告を完全に無効化
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
        # 乱数シードを設定して独立性を確保
        seed = int(time.time() * 1000000) % (2**32) + p_index
        np.random.seed(seed)
        
        # すべてのプロセス（メイン含む）で出力を抑制
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # メインプロセスも含めて全ての出力を抑制
        if p_index >= 0:  # 全てのプロセスで出力を抑制
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        
        try:
            result = cls.run_agent(episode_per_agent, weights, sigma)
        finally:
            # 出力を元に戻す（メインプロセスのみ）
            if p_index == 0:  # メインプロセスの場合のみ出力を戻す
                sys.stdout.close()
                sys.stdout = old_stdout
                sys.stderr.close()
                sys.stderr = old_stderr
            
        return result

    def train(self, epoch=100, episode_per_agent=1, render=False, silent=False):
        # GPU使用時は並列数を調整
        is_using_gpu = any('GPU' in device.name for device in tf.config.list_physical_devices())
        n_jobs = 4 if is_using_gpu else 7  # GPU使用時は並列数を減らす
        
        if not silent:
            print(f"{'GPU' if is_using_gpu else 'CPU'} モード: {n_jobs}並列で実行")

        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()
        
        # ここを修正: silentパラメータでメッセージを制御
        if not silent:
            print(f"Using {n_jobs} CPU cores for parallel processing")
        
        # 各エポックで並列処理を実行
        for e in range(epoch):
            # M3向け最適化
            parallel = Parallel(n_jobs=n_jobs, verbose=0, 
                              prefer="processes", # スレッドより高速なプロセスを使用
                              batch_size="auto", 
                              backend="multiprocessing", # より高速なバックエンド
                              max_nbytes=None) # メモリ制限を解除
            
            # パラメータを使用してクラスメソッドを呼び出す
            results = parallel(
                delayed(self.__class__.experiment)(
                    i, self.weights, self.sigma, episode_per_agent
                ) for i in range(self.population_size)
            )
                
            self.update(results)
            self.log()

        agent.model.set_weights(self.weights)
        return agent

    @classmethod
    def make_env(cls):
        return CartPoleVectorObserver()

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000):
        try:
            # 各実行で新しい環境を作成（並列処理でのリソース競合防止）
            env = cls.make_env()
            actions = list(range(env.action_space.n))
            agent = EvolutionalAgent(actions)

            noises = []
            new_weights = []

            # Make weight
            for w in base_weights:
                noise = np.random.randn(*w.shape)
                new_weights.append(w + sigma * noise)
                noises.append(noise)

            # Test Play
            total_reward = 0
            episode_count = 0  # 成功したエピソードのカウント
            for e in range(episode_per_agent):
                try:
                    s = env.reset()
                    if s is None:
                        continue  # リセットに失敗した場合は次のエピソードへ
                        
                    if agent.model is None:
                        agent.initialize(s, new_weights)
                    
                    done = False
                    step = 0
                    episode_reward = 0
                    
                    # バッチサイズを増やしてステップ数を減らす
                    BATCH_SIZE = 4  # 一度に複数のステップを評価
                    
                    while not done and step < max_step:
                        # 現在の状態を複製
                        states = np.array([s] * BATCH_SIZE) if BATCH_SIZE > 1 else np.array([s])
                        
                        # バッチ予測
                        actions = []
                        for state in states:
                            a = agent.policy(state)
                            actions.append(a)
                        
                        # 順次実行（環境は並列化できない）
                        for a in actions:
                            if done:
                                break
                            n_state, reward, done, info = env.step(a)
                            episode_reward += reward
                            s = n_state
                            step += 1

                    total_reward += episode_reward
                    episode_count += 1
                except Exception:
                    continue

            # 少なくとも1つのエピソードが成功した場合のみ平均を計算
            reward = total_reward / max(episode_count, 1)
            return reward, noises
        except Exception as e:
            return 0, [np.zeros_like(w) for w in base_weights]

    def update(self, agent_results):
        # フィルタリングを追加：エラーで戻り値が不正なものを除外
        valid_results = [r for r in agent_results if isinstance(r, tuple) and len(r) == 2 and r[0] is not None and r[1] is not None]
        
        if not valid_results:
            print("WARNING: No valid agent results to update weights")
            return
            
        rewards = np.array([r[0] for r in valid_results])
        
        # ノイズをリストとして扱う
        noises_list = []
        for r in valid_results:
            if isinstance(r[1], list) and len(r[1]) == len(self.weights):
                noises_list.append(r[1])
        
        if not noises_list:
            print("WARNING: No valid noise data found")
            return
            
        # 報酬の正規化
        if len(rewards) > 1:
            normalized_rs = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        else:
            normalized_rs = np.zeros_like(rewards)

        # Update base weights
        new_weights = []
        for i, w in enumerate(self.weights):
            try:
                # リストからi番目の重みのノイズだけを抽出
                noise_at_i = [n[i] for n in noises_list]
                
                # 各ノイズと正規化された報酬の積の合計を計算
                update_val = np.zeros_like(w)
                for ni, r in zip(noise_at_i, normalized_rs):
                    try:
                        update_val += ni * r
                    except Exception as e:
                        print(f"Error adding noise contribution: {e}")
                
                # 重みを更新
                rate = self.learning_rate / ((len(noises_list) + 1e-10) * self.sigma)
                w = w + rate * update_val
            except Exception as e:
                print(f"Weight update error at index {i}: {e}")
            new_weights.append(w)

        self.weights = new_weights
        self.reward_log.append(rewards)

    def log(self):
        rewards = self.reward_log[-1]
        print("Epoch {}: reward {:.3f}(max:{:.1f}, min:{:.1f})".format(
            len(self.reward_log), rewards.mean(),
            rewards.max(), rewards.min()))

    def plot_rewards(self):
        indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        plt.figure()
        plt.title("Reward History")
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="reward")
        plt.legend(loc="best")
        plt.show()


def main(play, epochs, pop_size, sigma, lr, silent=False):
    # サイレントモードの場合は余計な出力を最初から抑制
    if silent:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import warnings
        warnings.filterwarnings('ignore')

    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.h5")
    # 以下略...
    if play:
        env = EvolutionalTrainer.make_env()
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    else:
        trainer = EvolutionalTrainer(
            population_size=pop_size,
            sigma=sigma,
            learning_rate=lr,
            report_interval=5
        )
        trained = trainer.train(
            epoch=epochs,
            episode_per_agent=5,
            silent=silent  # silentフラグを渡す
        )
        trained.save(model_path)
        trainer.plot_rewards()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true", help="学習済みモデルで実行")
    parser.add_argument("--epochs", type=int, default=100, help="学習エポック数")
    parser.add_argument("--pop-size", type=int, default=100, help="集団サイズ")
    parser.add_argument("--sigma", type=float, default=0.1, help="探索のノイズ幅")
    parser.add_argument("--lr", type=float, default=0.05, help="学習率")
    parser.add_argument("--silent", action="store_true", help="警告を抑制")

    args = parser.parse_args()
    
    if args.silent:
        import warnings
        warnings.filterwarnings('ignore')
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # silentフラグに基づいてTensorFlow設定を構成
    configure_tensorflow(silent=args.silent)  # この1回だけ呼び出す
    
    main(args.play, args.epochs, args.pop_size, args.sigma, args.lr, args.silent)
