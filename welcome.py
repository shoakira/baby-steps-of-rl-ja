import numpy as np
from tensorflow.keras import models, layers, Input
import gymnasium as gym  # 最新バージョンではgymnasiumとして提供


def welcome():
    """
    基本的なライブラリのインストール確認と動作テストを行うコード
    簡単なニューラルネットワークを使って「CartPole」環境をプレイします
    """

    # 1. ゲーム環境の初期化 - 新APIではrender_modeを指定
    env = gym.make("CartPole-v1", render_mode="human")
    num_action = env.action_space.n  # 可能なアクション数を取得（左、右の2つ）
    episode_count = 10  # 実行するエピソード数

    # 2. 環境の初期状態を取得
    s, _ = env.reset()  # 新APIでは(obs, info)タプルを返す

    # 3. エージェント（ニューラルネットワーク）の構築 - 改善版
    brain = models.Sequential([
        Input(shape=(np.prod(s.shape),)),
        layers.Dense(num_action, activation="softmax")
    ])

    # 4. ポリシー（行動選択）関数の定義
    def policy(s):
        evaluation = brain(np.array([s.flatten()]), training=False)
        return np.argmax(evaluation)

    # 5. 複数エピソードの実行
    for e in range(episode_count):
        s, _ = env.reset()  # 新APIでは(obs, info)タプルを返す
        terminated = truncated = False  # 新APIではterminatedとtruncatedを使用
        
        # 6. エピソード内のステップ実行
        while not (terminated or truncated):
            a = policy(s)
            n_state, reward, terminated, truncated, info = env.step(a)  # 新APIでは5つの値を返す
            s = n_state

    env.close()  # 環境を適切に閉じる


if __name__ == "__main__":
    welcome()
