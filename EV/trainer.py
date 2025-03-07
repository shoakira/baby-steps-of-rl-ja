###############################################################################
# trainer.py - 進化戦略トレーナー
###############################################################################

import os
import sys
import time
import threading 
import numpy as np
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tensorflow as tf

# =============== ベース進化戦略トレーナー ===============
class EvolutionalTrainer:
    """進化戦略（ES）アルゴリズムによるエージェント訓練の基本クラス"""
    
    def __init__(self, population_size=20, sigma=0.5, learning_rate=0.1,
                 report_interval=10, timeout_seconds=90):
        """初期化"""
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.timeout_seconds = timeout_seconds
        self.weights = ()
        self.reward_log = []
        
    def train(self, epoch=100, episode_per_agent=1, render=False, silent=False):
        """メイン学習ループ"""
        from environment import CartPoleVectorObserver
        from agent import EvolutionalAgent
        from config import Config
        
        # 初期化と環境準備
        train_start = time.time()
        if not silent:
            self._print_training_start_info(epoch)
            
        # エージェント準備
        env = CartPoleVectorObserver()
        agent = self._prepare_agent(env)
        
        # 並列評価エンジン作成
        parallel = self._create_parallel_executor(Config.get_parallel_jobs())
        if not silent:
            print(f"[4/5] 学習開始: {epoch}エポック x {self.population_size}個体")
        
        # エポックループ
        for e in range(epoch):
            epoch_start = time.time()
            
            # 個体評価と重み更新
            self._run_single_epoch(e, epoch, parallel, episode_per_agent, train_start, silent)
            
        # 学習完了
        if not silent:
            print("[5/5] 学習完了、モデル最終化中...")
            
        agent.model.set_weights(self.weights)
        return agent
    
    def _print_training_start_info(self, epoch):
        """学習開始情報の表示"""
        print("===== 進化戦略学習を開始します =====")
        print(f"エポック数: {epoch}, 集団サイズ: {self.population_size}")
        print(f"探索パラメータ: σ={self.sigma}, 学習率={self.learning_rate}")
        
        # GPU/CPU検出と表示
        is_using_gpu = any('GPU' in d.name for d in tf.config.list_physical_devices())
        n_jobs = min(4, self._get_optimal_job_count())
        print(f"[1/5] 環境準備: {'GPU + ' if is_using_gpu else ''}CPU並列: {n_jobs}個")
        print("[2/5] モデル初期化中...")
    
    def _prepare_agent(self, env):
        """エージェントの初期化"""
        from agent import EvolutionalAgent
        
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()
        return agent
    
    def _create_parallel_executor(self, worker_count):
        """並列実行エンジンの作成"""
        print("[3/5] 並列評価エンジン準備中...")
        return Parallel(
            n_jobs=min(worker_count, 4),
            verbose=0,
            prefer="processes",
            batch_size=1,
            timeout=self.timeout_seconds * 1.5,
            backend="multiprocessing",
            max_nbytes=None
        )
    
    def _run_single_epoch(self, e, epoch, parallel, episode_per_agent, train_start, silent):
        """1エポックの実行"""
        # 進捗表示
        if not silent:
            elapsed = time.time() - train_start
            print(f"\r  エポック {e+1}/{epoch}: [1/2] 個体評価中... ({elapsed:.1f}秒)", end="")
            sys.stdout.flush()
        
        # 並列評価
        results = self._evaluate_population(parallel, episode_per_agent)
        
        # 重み更新
        if not silent:
            elapsed = time.time() - train_start
            print(f"\r  エポック {e+1}/{epoch}: [2/2] 重み更新中... ({elapsed:.1f}秒)", end="")
            sys.stdout.flush()
        
        self._update_weights(results)
        
        # 結果表示
        if not silent:
            rewards = self.reward_log[-1]
            print(f"\r  エポック {e+1}/{epoch}: 報酬 {rewards.mean():.1f} (最大:{rewards.max():.1f})")
    
    def _evaluate_population(self, parallel, episode_per_agent):
        """集団全体の並列評価"""
        return parallel(
            delayed(EvaluationWorker.evaluate)(
                i, self.weights, self.sigma, episode_per_agent, self.timeout_seconds
            ) for i in range(self.population_size)
        )
    
    def _update_weights(self, agent_results):
        """進化戦略による重み更新"""
        updater = WeightUpdater(self.weights, self.sigma, self.learning_rate)
        self.weights, rewards = updater.update(agent_results)
        self.reward_log.append(rewards)
    
    def _get_optimal_job_count(self):
        """最適な並列ジョブ数を取得"""
        from config import Config
        return Config.get_parallel_jobs()
    
    def plot_rewards(self, save_path=None, y_max=None):
        """学習曲線のプロット"""
        plotter = LearningCurvePlotter(
            self.reward_log, 
            self.population_size, 
            self.sigma, 
            self.learning_rate
        )
        return plotter.plot(save_path, y_max)


# =============== 並列評価ワーカー ===============
class EvaluationWorker:
    """個体評価を実行する並列ワーカー"""
    
    @classmethod
    def evaluate(cls, p_index, weights, sigma, episode_per_agent, timeout_seconds=90):
        """個体評価実験（並列処理用）"""
        from config import Config
        
        # 独立した乱数シード設定
        np.random.seed(int(time.time() * 1000000) % (2**32) + p_index)
        
        # タイムアウト機構セットアップ
        timer, timeout_handler = cls._setup_timeout(p_index, timeout_seconds)
        
        try:
            # 評価開始
            cls._log_progress(p_index, "評価中...")
            timer.start()
            
            # エージェント実行
            result, debug_info = cls._run_agent(
                episode_per_agent, weights, sigma, 
                Config.MAX_STEPS_PER_EPISODE, p_index
            )
            
            # タイマー停止と結果ログ
            timer.cancel()
            cls._log_progress(p_index, f"完了: {debug_info[:60]}...")
            return result
            
        except Exception as e:
            # エラー時はゼロ報酬を返す
            cls._log_progress(p_index, f"エラー: {str(e)[:30]}")
            return (0, [np.zeros_like(w) for w in weights])
            
        finally:
            # 確実にタイマーを停止
            timer.cancel()
    
    @classmethod
    def _setup_timeout(cls, p_index, timeout_seconds):
        """タイムアウト機構のセットアップ"""
        class TimeoutError(Exception):
            pass
            
        def timeout_handler():
            """タイムアウト発生時のコールバック"""
            cls._log_progress(p_index, "タイムアウト!")
            raise TimeoutError("評価タイムアウト")
        
        timer = threading.Timer(timeout_seconds, timeout_handler)
        return timer, timeout_handler
    
    @classmethod
    def _log_progress(cls, p_index, message):
        """プロセスの進捗表示（メインプロセスのみ）"""
        if p_index == 0:
            sys.stdout.write(f"\r      個体{p_index}: {message}")
            sys.stdout.flush()
    
    @classmethod
    def _run_agent(cls, episode_per_agent, weights, sigma, max_step, p_index):
        """エージェントによる評価実行"""
        from agent import EvolutionalAgent
        from environment import CartPoleVectorObserver
        
        # 環境とエージェント作成
        env = CartPoleVectorObserver()
        actions = list(range(env.action_space.n))
        agent = EvolutionalAgent(actions)
        
        # ノイズを生成して重みに加える
        noises, noisy_weights = cls._add_noise_to_weights(weights, sigma)
        
        # エージェント評価
        total_reward = 0
        episode_count = 0
        detailed_logs = []
        
        for e in range(episode_per_agent):
            # エピソード実行
            episode_reward, steps = cls._run_episode(
                env, agent, noisy_weights, max_step, p_index, e
            )
            
            if episode_reward is not None:
                detailed_logs.append(f"エピソード{e+1}: {steps}ステップ, 報酬{episode_reward:.1f}")
                total_reward += episode_reward
                episode_count += 1
        
        # 結果返却
        reward = total_reward / max(episode_count, 1)
        details = " | ".join(detailed_logs)
        return (reward, noises), details
    
    @classmethod
    def _add_noise_to_weights(cls, weights, sigma):
        """重みにノイズを加える"""
        noises = []
        noisy_weights = []
        
        for w in weights:
            noise = np.random.randn(*w.shape)
            noisy_weights.append(w + sigma * noise)
            noises.append(noise)
            
        return noises, noisy_weights
    
    @classmethod
    def _run_episode(cls, env, agent, noisy_weights, max_step, p_index, episode):
        """1エピソードを実行"""
        try:
            # 環境リセット
            s = env.reset()
            if s is None:
                return None, 0
            
            # エージェント初期化（初回のみ）
            if agent.model is None:
                agent.initialize(s, noisy_weights)
            
            # エピソード実行
            done = False
            step = 0
            episode_reward = 0
            
            while not done and step < max_step:
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
                step += 1
                
            return episode_reward, step
            
        except Exception as e:
            print(f"エピソード実行エラー: {e}")
            return None, 0


# =============== 重み更新クラス ===============
class WeightUpdater:
    """進化戦略による重み更新を行うクラス"""
    
    def __init__(self, weights, sigma, learning_rate):
        """初期化"""
        self.weights = weights
        self.sigma = sigma
        self.learning_rate = learning_rate
    
    def update(self, agent_results):
        """評価結果から重みを更新"""
        # 有効な結果のみフィルタリング
        valid_results = self._filter_valid_results(agent_results)
        
        if not valid_results:
            print("WARNING: 有効な評価結果がありません")
            return self.weights, np.array([0])
        
        # 報酬とノイズの抽出
        rewards, noises_list = self._extract_rewards_and_noises(valid_results)
        
        if not noises_list:
            print("WARNING: 有効なノイズデータがありません")
            return self.weights, rewards
        
        # 正規化された報酬の計算
        normalized_rewards = self._normalize_rewards(rewards)
        
        # 重みの更新
        new_weights = self._compute_new_weights(normalized_rewards, noises_list)
        
        return new_weights, rewards
    
    def _filter_valid_results(self, agent_results):
        """有効な結果をフィルタリング"""
        return [r for r in agent_results if isinstance(r, tuple) and len(r) == 2 
                and r[0] is not None and r[1] is not None]
    
    def _extract_rewards_and_noises(self, valid_results):
        """報酬とノイズを抽出"""
        rewards = np.array([r[0] for r in valid_results])
        
        noises_list = []
        for r in valid_results:
            if isinstance(r[1], list) and len(r[1]) == len(self.weights):
                noises_list.append(r[1])
                
        return rewards, noises_list
    
    def _normalize_rewards(self, rewards):
        """報酬を正規化（平均0、標準偏差1）"""
        if len(rewards) > 1:
            return (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        else:
            return np.zeros_like(rewards)
    
    def _compute_new_weights(self, normalized_rewards, noises_list):
        """新しい重みを計算"""
        new_weights = []
        
        for i, w in enumerate(self.weights):
            try:
                # 更新値の計算
                update_val = self._calculate_update(i, w, normalized_rewards, noises_list)
                
                # 勾配的な更新式
                rate = self.learning_rate / ((len(noises_list) + 1e-10) * self.sigma)
                new_weights.append(w + rate * update_val)
                
            except Exception as e:
                print(f"重み更新エラー (層 {i}): {e}")
                new_weights.append(w)
                
        return new_weights
    
    def _calculate_update(self, layer_idx, weight, normalized_rewards, noises_list):
        """層ごとの更新値を計算"""
        update_val = np.zeros_like(weight)
        
        # 各ノイズ × 正規化報酬 の積和計算
        for noise, reward in zip([n[layer_idx] for n in noises_list], normalized_rewards):
            try:
                update_val += noise * reward
            except Exception:
                pass
                
        return update_val


# =============== 学習曲線プロッター ===============
class LearningCurvePlotter:
    """学習結果の可視化を行うクラス"""
    
    def __init__(self, reward_log, population_size, sigma, learning_rate):
        """初期化"""
        self.reward_log = reward_log
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
    
    def plot(self, save_path=None, y_max=None):
        """学習曲線のプロット"""
        from config import Config
        
        if not self.reward_log:
            return None
            
        # データ準備
        indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        
        # プロット作成
        self._create_plot(indices, means, stds, y_max)
        
        # ファイル保存
        return self._save_plot(save_path, Config.get_plot_dir())
    
    def _create_plot(self, indices, means, stds, y_max):
        """プロットの作成"""
        plt.figure(figsize=(10, 6))
        plt.title(f"Evolution Strategy Learning Curve (Pop={self.population_size}, σ={self.sigma}, LR={self.learning_rate})")
        plt.grid()
        
        # データプロット
        plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g", label=f"Reward (Final: {means[-1]:.1f})")
        
        # 軸ラベル
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.legend(loc="best")
        
        # Y軸設定
        max_reward = means.max() + stds.max()
        if y_max is None:
            if max_reward < 450:
                plt.ylim(0, max_reward * 1.1)
            else:
                plt.ylim(0, 550)
        else:
            plt.ylim(0, y_max)
        
        # 500報酬の線（解決ライン）
        if max_reward > 100 or y_max is None or y_max >= 500:
            plt.axhline(y=500, color='r', linestyle='--', alpha=0.3, label="Solved")
    
    def _save_plot(self, save_path, plot_dir):
        """プロットの保存"""
        if save_path:
            filename = os.path.basename(save_path)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"es_results_{timestamp}.png"
            
        final_path = os.path.join(plot_dir, filename)
        plt.savefig(final_path, dpi=100, bbox_inches='tight')
        print(f"学習曲線を保存しました: {final_path}")
        
        plt.show()
        return final_path