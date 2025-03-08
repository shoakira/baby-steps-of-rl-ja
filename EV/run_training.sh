#!/bin/bash

# 進化戦略学習の高効率実行スクリプト
echo "===== 進化戦略学習を開始します ====="
echo "システム状態: $(date)"
echo "CPU使用率: $(top -l 1 | grep "CPU usage" | awk '{print $3}')"
echo "メモリ空き容量: $(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')"

# システム設定最適化
echo "システム最適化中..."
#osascript -e 'tell application "Google Chrome" to quit' 2>/dev/null || true
#osascript -e 'tell application "Safari" to quit' 2>/dev/null || true

# ========== CPU専用モード強制設定 ==========
# GPUを無効化する環境変数
export CUDA_VISIBLE_DEVICES="-1"  # CUDA GPUを無効化
export TF_DISABLE_GPU="1"         # TensorFlowのGPU処理を無効化
export DISABLE_MPS="1"            # Apple Silicon MPSバックエンドを無効化
export TF_CPP_MIN_LOG_LEVEL="2"   # 警告を減らす

# CPU最適化設定
# 使用可能なCPUコア数を検出
CPU_COUNT=$(sysctl -n hw.ncpu)
echo "検出されたCPUコア数: ${CPU_COUNT}"
export OMP_NUM_THREADS="${CPU_COUNT}"  # OpenMP並列スレッド数
export MKL_NUM_THREADS="${CPU_COUNT}"  # Intel MKL並列スレッド数

echo "CPU専用モードを強制設定しました"
# ==========================================

# 進化戦略学習の実行
# Caffeinate: システムスリープを防止
# nice -n -10: 高い優先度でプロセスを実行
# Python最適化フラグ付き実行
caffeinate -i python main.py --epochs 50

echo "===== 学習完了 ====="
echo "終了時刻: $(date)"

# 完了通知（オプション）
osascript -e 'display notification "進化戦略学習が完了しました" with title "学習完了"'