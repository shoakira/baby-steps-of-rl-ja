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
        # 最新のKerasに対応
        normal = K.initializers.GlorotNormal()
        inputs = K.Input(shape=state.shape)
        x = K.layers.Conv2D(3, kernel_size=5, strides=3, 
                           kernel_initializer=normal, activation="relu")(inputs)
        x = K.layers.Flatten()(x)
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

        # 並列処理をシリアル処理に変更
        for e in range(epoch):
            results = []
            for p in range(self.population_size):
                result = EvolutionalTrainer.run_agent(
                    episode_per_agent, self.weights, self.sigma)
                results.append(result)
            self.update(results)
            self.log()

        agent.model.set_weights(self.weights)
        return agent

    @classmethod
    def make_env(cls):
        return CartPoleObserver(width=50, height=50, frame_count=1)

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000):
        try:
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
            for e in range(episode_per_agent):
                s = env.reset()
                if s is None:
                    return 0, noises
                    
                if agent.model is None:
                    agent.initialize(s, new_weights)
                
                done = False
                step = 0
                
                while not done and step < max_step:
                    a = agent.policy(s)
                    try:
                        n_state, reward, done, info = env.step(a)
                        if n_state is None:
                            done = True
                            continue
                        
                        total_reward += reward
                        s = n_state
                    except Exception as e:
                        print(f"Step execution error: {e}")
                        done = True
                    step += 1

            reward = total_reward / max(episode_per_agent, 1)
            return reward, noises
        except Exception as e:
            print(f"Agent execution error: {e}")
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
        trainer = EvolutionalTrainer()
        trained = trainer.train()
        trained.save(model_path)
        trainer.plot_rewards()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
