#!/bin/bash

# 進化戦略学習の高効率実行スクリプト
echo "===== 進化戦略学習を開始します ====="
echo "システム状態: $(date)"
echo "CPU使用率: $(top -l 1 | grep "CPU usage" | awk '{print $3}')"
echo "メモリ空き容量: $(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')"

# システム設定最適化
echo "システム最適化中..."
osascript -e 'tell application "Google Chrome" to quit' 2>/dev/null || true
osascript -e 'tell application "Safari" to quit' 2>/dev/null || true

# Caffeinate: システムスリープを防止
# nice -n -10: 高い優先度でプロセスを実行
# Python最適化フラグ付き実行
caffeinate -i nice -n -10 python evolution.py --epochs 200 --pop-size 120 --sigma 0.1 --silent

echo "===== 学習完了 ====="
echo "終了時刻: $(date)"

# 完了通知（オプション）
osascript -e 'display notification "進化戦略学習が完了しました" with title "学習完了"'

def train(self, epoch=100, episode_per_agent=1, render=False, silent=False):
    """進化戦略によるエージェント訓練（メイン処理）"""
    # GPU/CPU検出と並列数決定
    is_using_gpu = any('GPU' in device.name for device in tf.config.list_physical_devices())
    n_jobs = 4 if is_using_gpu else 7
    
    if not silent:
        print(f"{'GPU計算 + ' if is_using_gpu else ''}CPU並列処理: {n_jobs}個のプロセスで評価")

    # 環境と基本エージェントの準備
    env = self.make_env()
    actions = list(range(env.action_space.n))
    s = env.reset()
    agent = EvolutionalAgent(actions)
    agent.initialize(s)
    self.weights = agent.model.get_weights()
    
    # 並列処理設定（一度だけ初期化して再利用）
    parallel = Parallel(n_jobs=n_jobs, verbose=0, 
                      prefer="processes",
                      batch_size="auto",
                      backend="multiprocessing",
                      max_nbytes=None)
    
    # エポックごとの学習ループ
    for e in range(epoch):
        # 並列個体評価
        results = parallel(
            delayed(self.__class__.experiment)(
                i, self.weights, self.sigma, episode_per_agent
            ) for i in range(self.population_size)
        )
        
        # 結果を使って重みを更新
        self.update(results)
        self.log()

    # ここに追加: 訓練済みエージェントを返す
    agent.model.set_weights(self.weights)
    return agent