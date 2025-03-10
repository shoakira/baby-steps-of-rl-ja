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
                 report_interval=10, timeout_seconds=None):
        """初期化
        
        Args:
            population_size (int): 1世代あたりの個体数
            sigma (float): ノイズの標準偏差
            learning_rate (float): 更新率
            report_interval (int): 報告間隔
            timeout_seconds (float, optional): タイムアウト時間（指定がなければConfigから取得）
        """
        from config import Config
        
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        # タイムアウトはConfigから取得（オーバーライド可能）
        self.timeout_seconds = timeout_seconds or Config.get_evaluation_timeout()
        self.weights = ()
        self.reward_log = []
        
    def train(self, epoch=100, episode_per_agent=1, render=False, silent=False, realtime_plot=False):
        """メイン学習ループ
        
        Args:
            epoch: 学習エポック数
            episode_per_agent: エージェントあたりのエピソード数
            render: レンダリングを行うか
            silent: 出力を抑制するか
            realtime_plot: リアルタイムプロットを表示するか
        """
        from environment import CartPoleVectorObserver
        from agent import EvolutionalAgent
        from config import Config
        
        # リアルタイムプロッター初期化
        plotter = None
        if realtime_plot and not silent:
            from utils import RealtimePlotter
            plotter = RealtimePlotter(title=f"Evolution Strategy (Pop={self.population_size}, σ={self.sigma}, η={self.learning_rate})")
        
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
            
            # リアルタイムプロット更新
            if plotter and len(self.reward_log) > 0:
                plotter.update(e+1, self.reward_log[-1])
        
        # 学習完了
        if not silent:
            print("[5/5] 学習完了、モデル最終化中...")
            
        agent.model.set_weights(self.weights)
        
        # プロット保存
        if plotter:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            plotter.save(f"es_realtime_e{epoch}_p{self.population_size}_s{self.sigma}_lr{self.learning_rate}_{timestamp}.png")
            
        return agent
    
    def _get_device_info(self):
        """実行デバイス情報を取得
        
        Returns:
            tuple: (GPU使用フラグ, 並列ジョブ数, デバイス詳細情報)
        """
        # GPU/CPU検出
        gpus = tf.config.list_physical_devices('GPU')
        mps_device = any('MPS' in d.name for d in tf.config.list_physical_devices())
        is_using_gpu = len(gpus) > 0 or mps_device
        
        # 並列ジョブ数
        n_jobs = self._get_optimal_job_count()
        
        # デバイス詳細
        device_info = ""
        if is_using_gpu:
            if mps_device:
                device_info = "Apple Silicon GPU (MPS)"
            else:
                device_info = f"NVIDIA GPU ({len(gpus)}基)"
        else:
            device_info = f"CPU ({n_jobs}コア)"
        
        return is_using_gpu, n_jobs, device_info

    def _print_training_start_info(self, epoch):
        """学習開始情報の表示"""
        print("===== 進化戦略学習を開始します =====")
        print(f"エポック数: {epoch}, 集団サイズ: {self.population_size}")
        print(f"探索パラメータ: σ={self.sigma}, 学習率={self.learning_rate}")
        
        # GPU/CPU検出と表示
        is_using_gpu, n_jobs, device_info = self._get_device_info()
        print(f"[1/5] 環境準備: {device_info} で並列処理: {n_jobs}個")
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
        """並列実行エンジンの作成
        
        Args:
            worker_count: Configから取得した最適な並列プロセス数
            
        Returns:
            Parallel: 並列処理エンジン
        """
        print(f"[3/5] 並列評価エンジン準備中... ({worker_count}コア)")
        
        return Parallel(
            n_jobs=worker_count,  # 並列度をより多く
            verbose=0,
            prefer="threads",     # スレッドベースの並列処理
            timeout=self.timeout_seconds,
            backend="threading",  # スレッドバックエンド
            max_nbytes="1M"       # メモリ制限
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
        """集団の評価を並列実行（重みのみ送信）
        
        各プロセスで独立してエージェントを作成し、pickle化の問題を回避します
        
        Args:
            parallel: 並列実行エンジン
            episode_per_agent: 個体あたりエピソード数
            
        Returns:
            評価結果リスト [(報酬リスト, 個体インデックス), ...]
        """
        # 現在のベース重みを取得
        base_weights = self.weights
        
        # 重みとシグマだけを渡して並列評価
        return parallel(
            delayed(EvaluationWorker.evaluate_weights)(
                p_index, episode_per_agent, base_weights, self.sigma, self.timeout_seconds
            )
            for p_index in range(self.population_size)
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

    def _get_population_weights(self):
        """集団の個体（重み）を生成
        
        各個体は同じベース重みから始まり、評価時にノイズが加えられます
        
        Returns:
            list: 個体（重み）のリスト
        """
        population = []
        for _ in range(self.population_size):
            population.append(self.weights)
        return population


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
        
        timer = threading.Timer(timeout_seconds * 1.5, timeout_handler)
        return timer, timeout_handler
    
    # _run_single_epoch メソッド内で EvaluationWorker の初期化時にフラグ設定

    @classmethod
    def evaluate(cls, p_index, episode_per_agent, agent, weights, env, timeout_seconds):
        """パラメーターセットの評価
        
        Args:
            p_index: 並列処理インデックス
            episode_per_agent: 個体あたりエピソード数
            agent: エージェントインスタンス
            weights: 評価する重み
            env: 環境
            timeout_seconds: タイムアウト時間
            
        Returns:
            (報酬リスト, 個体インデックス)
        """
        # タイムアウト機構をセットアップ
        timer, timeout_handler = cls._setup_timeout(p_index, timeout_seconds)
        timer.start()
        
        # 並列処理時のログ制御用フラグ
        agent.is_main_process = (p_index == 0)
        
        # 以下は既存の処理...

    # 3. EvaluationWorker._log_progress の改善
    @classmethod
    def _log_progress(cls, p_index, message):
        """プロセスの進捗表示
        
        並列処理時には標準出力が競合するため、メインプロセス(p_index=0)の
        メッセージのみを表示します。これにより処理状況を確認しつつ、
        ログの混乱を防止します。
        
        Args:
            p_index (int): プロセスインデックス (0がメインプロセス)
            message (str): 表示するメッセージ
        """
        if p_index == 0:  # メインプロセスのみ出力
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

    @classmethod
    def evaluate_weights(cls, p_index, episode_per_agent, base_weights, sigma, timeout_seconds):
        """重みのみを受け取って評価する方法
        
        各プロセスで新たにエージェントを作成するため、pickle化問題を回避できます
        
        Args:
            p_index: 並列処理インデックス
            episode_per_agent: 個体あたりエピソード数
            base_weights: ベース重み
            sigma: ノイズ幅
            timeout_seconds: タイムアウト時間
        """
        from agent import EvolutionalAgent
        from environment import CartPoleVectorObserver
        from config import Config
        
        # 独立した乱数シード設定
        np.random.seed(int(time.time() * 1000000) % (2**32) + p_index)
        
        # タイムアウト機構
        timer, timeout_handler = cls._setup_timeout(p_index, timeout_seconds)
        
        try:
            # ログ表示
            cls._log_progress(p_index, "評価中...")
            timer.start()
            
            # ノイズを生成して重みに加える
            noises = []
            noisy_weights = []
            for w in base_weights:
                noise = np.random.randn(*w.shape)
                noisy_weights.append(w + sigma * noise)
                noises.append(noise)
            
            # 環境とエージェント作成（各プロセスで独立）
            env = CartPoleVectorObserver()
            agent = EvolutionalAgent(list(range(env.action_space.n)))
            s = env.reset()
            
            # ノイズ付き重みでエージェント初期化
            agent.is_main_process = (p_index == 0)  # メインプロセスフラグ
            agent.initialize(s, weights=noisy_weights)
            
            # エピソード実行
            total_reward = 0
            episode_count = 0
            detailed_logs = []
            
            for e in range(episode_per_agent):
                s = env.reset()
                done = False
                step = 0
                episode_reward = 0
                
                while not done and step < Config.MAX_STEPS_PER_EPISODE:
                    a = agent.policy(s)
                    n_state, reward, done, info = env.step(a)
                    episode_reward += reward
                    s = n_state
                    step += 1
                
                detailed_logs.append(f"エピソード{e+1}: {step}ステップ, 報酬{episode_reward:.1f}")
                total_reward += episode_reward
                episode_count += 1
            
            # 結果計算
            avg_reward = total_reward / max(episode_count, 1)
            details = " | ".join(detailed_logs)
            
            timer.cancel()
            cls._log_progress(p_index, f"完了: {details[:60]}...")
            
            # 修正: 二重タプルではなく単一のタプルを返す
            return avg_reward, noises  # detailsは返さない
            
        except Exception as e:
            cls._log_progress(p_index, f"エラー: {str(e)[:30]}")
            timer.cancel()
            return 0, [np.zeros_like(w) for w in base_weights]


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