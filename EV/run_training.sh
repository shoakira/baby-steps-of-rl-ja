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
caffeinate -i nice -n -10 python evolution.py --epochs 15 --pop-size 120 --sigma 0.1 --silent

echo "===== 学習完了 ====="
echo "終了時刻: $(date)"

# 完了通知（オプション）
osascript -e 'display notification "進化戦略学習が完了しました" with title "学習完了"'