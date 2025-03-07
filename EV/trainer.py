###############################################################################
# trainer.py - 進化戦略トレーナー
###############################################################################

import os
import sys
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tensorflow as tf

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
        import threading
        import signal
        
        # プロセスごとに独立した乱数シード設定
        seed = int(time.time() * 1000000) % (2**32) + p_index
        np.random.seed(seed)
        
        # メインプロセス（index=0）のみ簡潔な進捗表示
        if p_index == 0:
            sys.stdout.write(f"\r      個体{p_index}: エピソード評価中...\n")  # 改行を追加してバッファフラッシュを促進
            sys.stdout.flush()
        
        # タイムアウト機構
        class TimeoutError(Exception):
            pass
            
        def timeout_handler():
            if p_index == 0:
                sys.stdout.write(f"\r      個体{p_index}: タイムアウトしました。スキップします。\n")
                sys.stdout.flush()
            raise TimeoutError("個体評価がタイムアウトしました")
        
        # タイムアウトタイマーをセット（30秒）
        timer = threading.Timer(30.0, timeout_handler)
        
        try:
            # タイマー開始
            timer.start()
            
            # エージェント実行で報酬とノイズを取得
            result, debug_info = cls.run_agent(episode_per_agent, weights, sigma, max_step=Config.MAX_STEPS_PER_EPISODE, p_index=p_index)
            
            # タイマーキャンセル
            timer.cancel()
            
            # メインプロセスの場合、デバッグ情報を表示
            if p_index == 0 and debug_info:
                sys.stdout.write(f"\r      評価完了: {debug_info[:60]}...\n")  # 改行を追加
                sys.stdout.flush()
                
            return result
                    
        except TimeoutError:
            # タイムアウト時はゼロ報酬を返す
            return (0, [np.zeros_like(w) for w in weights])
                    
        except Exception as e:
            # タイマーキャンセル
            timer.cancel()
            
            # エラー時はゼロ報酬を返す
            if p_index == 0:
                sys.stdout.write(f"\r      エラー: {str(e)[:30]}...\n")  # 改行を追加
                sys.stdout.flush()
            return (0, [np.zeros_like(w) for w in weights])
        
        finally:
            # 確実にタイマーをキャンセル
            timer.cancel()

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
                          batch_size=1,  # バッチサイズを1に設定して、処理の詰まりを防止
                          timeout=45,    # 45秒のグローバルタイムアウトを設定
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
                print(f"\r  エポック {e+1}/{epoch}: [タスク 1/2] 個体評価中... (経過時間: {elapsed:.1f}秒)")  # 改行を追加
                sys.stdout.flush()
            
            # バッチサイズを小さくして並列処理の負荷分散を改善
            results = parallel(
                delayed(self.__class__.experiment)(
                    i, self.weights, self.sigma, episode_per_agent
                ) for i in range(self.population_size)
            )
            
            # 重み更新
            if not silent:
                elapsed = time.time() - train_start
                print(f"\r  エポック {e+1}/{epoch}: [タスク 2/2] 重み更新中... (経過時間: {elapsed:.1f}秒)")  # 改行を追加
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
                            
                            # 進捗状況をより明確に示すため、50ステップごとに改行を入れる
                            if step % 50 == 0 and step > 0:
                                print(f"\n        ... ステップ{step}実行中 (報酬:{episode_reward:.1f}) ...")
                        
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
