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
import os, sys
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
            s, _ = env.reset()  # 環境リセット
            done = False
            terminated = False
            truncated = False
            episode_reward = 0
            
            # エピソード実行
            while not done:
                if render:
                    env.render()  # 可視化
                a = self.policy(s)  # 行動選択
                n_state, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
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
        import time
        import numpy as np
        import os
        
        # 並列処理のデータ転送を最適化（新規追加）
        import pickle
        pickle.HIGHEST_PROTOCOL = 4  # シリアライズを高速化
        
        # メインプロセスのみ進捗を表示（joblib内部で処理されるため通常は表示されない）
        if p_index == 0:
            print(f"\r    個体評価: 0/{episode_per_agent}エピソード", end="")
        
        # TensorFlow警告抑制
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
        # プロセスごとに独立した乱数シード設定
        seed = int(time.time() * 1000000) % (2**32) + p_index
        np.random.seed(seed)
        
        # 並列プロセスの出力リダイレクト（ログ混在防止）
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # 全プロセスの出力を抑制
        if p_index >= 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        
        try:
            # エージェント実行で報酬とノイズを取得
            result = cls.run_agent(episode_per_agent, weights, sigma)
        finally:
            # メインプロセスの出力を元に戻す
            if p_index == 0:  # インデックス0のみ
                sys.stdout.close()
                sys.stdout = old_stdout
                sys.stderr.close()
                sys.stderr = old_stderr
            
        return result

    def train(self, epoch=100, episode_per_agent=1, render=False, silent=False):
        """進化戦略によるエージェント訓練（メイン処理）"""
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
            if not silent:
                print(f"\r  エポック {e+1}/{epoch}: 評価中...", end="")
                sys.stdout.flush()
                
            # 並列個体評価
            results = parallel(
                delayed(self.__class__.experiment)(
                    i, self.weights, self.sigma, episode_per_agent
                ) for i in range(self.population_size)
            )
            
            # 結果を使って重みを更新
            self.update(results)
            
            if not silent:
                rewards = self.reward_log[-1]
                print(f"\r  エポック {e+1}/{epoch}: 報酬 {rewards.mean():.1f} (最大:{rewards.max():.1f})")

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
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000):
        """エージェント実行による評価（個体の適合度計算）"""
        try:
            # 各実行で新しい環境作成（並列処理でのリソース競合防止）
            env = cls.make_env()
            actions = list(range(env.action_space.n))
            agent = EvolutionalAgent(actions)

            # ノイズ付き重みベクトル生成
            noises = []  # 適用したノイズを記録
            new_weights = []  # ノイズ適用済み重み
            for w in base_weights:
                noise = np.random.randn(*w.shape)  # ガウス分布ノイズ
                new_weights.append(w + sigma * noise)  # ノイズ適用
                noises.append(noise)  # ノイズ記録（勾配計算用）

            # エージェント評価
            total_reward = 0
            episode_count = 0  # 完了エピソード数
            for e in range(episode_per_agent):
                try:
                    s = env.reset()
                    if s is None:
                        continue  # リセット失敗時はスキップ
                        
                    # エージェント初期化（初回のみ）
                    if agent.model is None:
                        agent.initialize(s, new_weights)
                    
                    done = False
                    step = 0
                    episode_reward = 0
                    
                    # バッチ処理のサイズを動的に調整（環境に応じて）
                    import psutil
                    avail_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                    BATCH_SIZE = 8 if avail_mem > 4 else 4  # 利用可能メモリに基づく調整
                    
                    # エピソード実行
                    while not done and step < max_step:
                        # 状態のバッチ化（同じ状態を複製）
                        states = np.array([s] * BATCH_SIZE) if BATCH_SIZE > 1 else np.array([s])
                        
                        # バッチ予測で行動を事前計算
                        actions = []
                        for state in states:
                            a = agent.policy(state)
                            actions.append(a)
                        
                        # 環境ステップ実行（環境は並列化できないので順次処理）
                        for a in actions:
                            if done:
                                break
                            n_state, reward, done, info = env.step(a)
                            episode_reward += reward
                            s = n_state
                            step += 1

                    # エピソード完了
                    total_reward += episode_reward
                    episode_count += 1
                except Exception:
                    continue  # エラー時は次のエピソードへ

            # 平均報酬計算（完了エピソードのみ）
            reward = total_reward / max(episode_count, 1)
            return reward, noises  # 報酬とノイズを返す
        except Exception as e:
            # 全体エラー時はゼロ報酬を返す
            return 0, [np.zeros_like(w) for w in base_weights]

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

    def plot_rewards(self, save_path=None):
        """学習曲線のプロット（ファイル保存機能追加）"""
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
        
        # 500報酬の線を追加（CartPoleのゴール）
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
        env = EvolutionalTrainer.make_env()
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
            
        # 結果の保存と表示（plotfileフォルダに保存）
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        result_filename = f"es_results_e{epochs}_p{pop_size}_s{sigma}_lr{lr}_{timestamp}.png"
        # パスを直接指定せず、plot_rewards内でフォルダを処理する
        trainer.plot_rewards(result_filename)


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
