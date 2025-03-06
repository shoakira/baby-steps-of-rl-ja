import os
import argparse
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
import matplotlib.pyplot as plt
import gymnasium as gym  # gymからgymnasiumに変更

# Disable TensorFlow GPU for parallel execution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.keras as K

class EvolutionalAgent():

    def __init__(self, actions):
        self.actions = actions
        self.model = None

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        return agent

    def initialize(self, state, weights=()):
        # シンプルなフィードフォワードネットワーク
        normal = K.initializers.GlorotNormal()
        inputs = K.Input(shape=(len(state),))  # 状態ベクトルの次元
        x = K.layers.Dense(24, activation="relu", kernel_initializer=normal)(inputs)
        x = K.layers.Dense(24, activation="relu", kernel_initializer=normal)(x)
        outputs = K.layers.Dense(len(self.actions), activation="softmax")(x)
        model = K.Model(inputs=inputs, outputs=outputs)
        self.model = model
        
        if len(weights) > 0:
            self.model.set_weights(weights)

    def policy(self, state):
        state_array = np.array([state])
        # 最新のTensorFlowに対応
        action_probs = self.model.predict(state_array, verbose=0)[0]
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
        self._env = gym.make("CartPole-v1")
    
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

    def train(self, epoch=100, episode_per_agent=1, render=False):
        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()

        # CPUコア数を取得してMac向けに最適化
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count() - 1, 8)  # Macの場合は適切な値を設定
        if n_jobs <= 0:
            n_jobs = 1
        print(f"Using {n_jobs} CPU cores for parallel processing")
        
        # 各エポックで並列処理を実行
        for e in range(epoch):
            # 関数オブジェクト作成（各プロセスで実行する処理）
            def experiment(p_index):
                seed = np.random.randint(0, 2**32)
                np.random.seed(seed)
                return EvolutionalTrainer.run_agent(
                    episode_per_agent, self.weights, self.sigma)
            
            # 並列処理の実行
            parallel = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")
            results = parallel(delayed(experiment)(i) for i in range(self.population_size))
                
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
                    
                    while not done and step < max_step:
                        a = agent.policy(s)
                        try:
                            n_state, reward, done, info = env.step(a)
                            if n_state is None:
                                break
                            
                            episode_reward += reward
                            s = n_state
                        except Exception as e:
                            break
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


def main(play):
    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.h5")

    if play:
        env = EvolutionalTrainer.make_env()
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    else:
        trainer = EvolutionalTrainer(
            population_size=100,  # より多くの個体で探索
            sigma=0.1,            # ノイズの大きさを調整
            learning_rate=0.05,   # 学習率を調整
            report_interval=5
        )
        trained = trainer.train(
            epoch=200,            # より多くのエポックで学習
            episode_per_agent=5   # 各エージェントをより多くのエピソードで評価
        )
        trained.save(model_path)
        trainer.plot_rewards()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
