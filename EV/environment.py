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
