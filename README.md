# DistanceEstimationML

2 つの入力特徴量 under_y・theta から距離 distance(cm) を推定する機械学習（多項式回帰）プロジェクトです。scikit-learn で学習・評価・可視化を行い、最適な次数のモデルを保存します。学習済みモデルをマイコン（Teensy 4.1）へ移植した C++ 実装も同梱しています。

## 主な機能
- 多項式特徴量の生成と標準化（StandardScaler）
- 1〜20 次の多項式回帰の一括学習・評価（MAE/RMSE）
- 次数ごとの学習/検証 MAE の比較プロット
- 予測 vs 実測・残差プロットの保存
- ベストモデルの選択と `.joblib` 保存（係数・スケーラー含む）
- Teensy 4.1 向けの高精度（double）推論コード（17 次の例）

## リポジトリ構成
- `train.py`: 学習・評価・可視化をまとめて実行
- `utils.py`: データ読み込み・前処理・特徴量生成・補助関数
- `visualization.py`: 可視化ユーティリティ（MAE 曲線、散布図、残差）
- `requirements.txt`: Python 依存パッケージ
- `data/`: 入力 CSV サンプル（ヘッダ: `under_y, theta, distance`）
- `models/`: 学習済みモデル（`.joblib`）がタイムスタンプごとに保存
- `results/`: 可視化画像や MAE 結果 CSV がタイムスタンプごとに保存
- `teensy_distance_predictor_degree17/`: Teensy 4.1 向け推論コード（C++）

## データ要件（CSV）
`utils.load_data()` は以下の 3 列を前提に読み込みます。
- `under_y` (float)
- `theta` (float)
- `distance` (float; 目的変数)

サンプルは `data/` に複数あります。独自データを使う場合はヘッダ名を上記に合わせてください。

## セットアップ
- 推奨: Python 3.11.x

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依存パッケージ（抜粋）: numpy, scikit-learn, pandas, matplotlib, seaborn, joblib

## 使い方（学習〜評価）
1. `train.py` のデータパスを設定します。
   - 変数 `train_path` / `val_path` に学習/検証 CSV を指定します。
   - 同一ファイルを指定した場合、内部で 7:3 に分割します（`train_test_split`）。
   - `train_sub_ratio` を 0.0〜1.0 で指定すると、学習に使うサンプル割合を絞れます。

2. 学習を実行します。
```zsh
python train.py
```

3. 生成物を確認します（タイムスタンプは実行時に作成）。
- `models/YYYYMMDD_HHMMSS/`
  - `polynomial_degree{d}_mae{val_mae}.joblib`（係数、スケーラー、指標を内包）
- `results/YYYYMMDD_HHMMSS/`
  - `degree_vs_mae.png`（次数 vs MAE）
  - `predictions_degree_{d}.png`（訓練 予測 vs 実測）
  - `residuals_degree_{d}.png`（訓練 残差プロット）
  - `validation_predictions_best_model.png` / `validation_residuals_best_model.png`
  - `mae_results.csv`（次数ごとの検証 MAE）

4. ベストモデルの基準
- 検証 MAE が最小の次数をベストとして選択・報告します。

注意: 現在の `train.py` の初期値は存在しないファイル名が入っている可能性があります。`data/` 配下の実在ファイル（例: `All measurement data.csv`）に合わせて適宜変更してください。

## 実装メモ
- 特徴量: `sklearn.preprocessing.PolynomialFeatures` による全項（バイアス含む）
- スケーリング: `StandardScaler` を多項式後の行列に適用
- 学習器: `LinearRegression`
- 指標: MAE/RMSE（学習・検証の両方を出力）
- 保存名: `utils.format_filename(degree, mae)` により `polynomial_degree{d}_mae{:.2f}.joblib`
- 同一 CSV を学習/検証に指定した場合は 7:3 分割（固定 `random_state=42`）

## Teensy 4.1 へのデプロイ（サンプル: 17 次）
フォルダ `teensy_distance_predictor_degree17/` に、学習済み 17 次モデルを移植した C++ 実装が含まれます（ダブル精度）。

- 主なファイル
  - `teensy_distance_predictor_degree17.ino`: シリアルメニューで予測・ベンチマーク・PC 版比較などが可能
  - `teensy_polynomial_model.h/.cpp`: 係数・スケーラー（平均/スケール）・推論パイプライン
- 想定環境: Teensy 4.1（ARM Cortex-M7, FPU 有効）
- 入力レンジ: `UNDER_Y_MIN..MAX`=`-100..100`, `THETA_MIN..MAX`=`-180..180`
- メニュー例（シリアル 115200bps）:
  - `1`: 単一予測（手動入力）
  - `2`: 事前定義テストスイート
  - `3`: ベンチマーク（反復計測）
  - `4`: 連続モード ON/OFF
  - `5`: メモリ情報
  - `6`: 統計リセット
  - `7`: ストレステスト
  - `8`: PC 版との精度比較（詳細ログ）

ビルド手順（例）:
1. Arduino IDE + Teensyduino をインストール
2. ボードに Teensy 4.1 を選択、最適化 O2（デフォルト相当）
3. フォルダを開いてビルド/書き込み

注意:
- 本 C++ 実装は特定の `.joblib` から自動生成された係数/スケーラーを静的に埋め込んでいます。
- 別モデルへ更新したい場合はエクスポートスクリプトが必要です。（現状このリポジトリには未同梱）

## トラブルシューティング
- データが読めない / 列が足りない
  - CSV に `under_y, theta, distance` の 3 列があるか確認
- 結果が保存されない
  - `results/`・`models/` の作成は自動ですが、権限/パスを確認
- プロット生成でエラー
  - サーバー環境でも `savefig()` のみを用いており、GUI は不要です
- 数値が発散する/極端
  - 多項式の次数が高いと不安定になり得ます。`degree` 範囲を調整してください
- Teensy で不正な入力
  - 指定レンジ外ではガードでエラー値（-1）を返します

## よく使う変更ポイント
- 学習に使うデータ割合: `train_sub_ratio`
- 次数レンジ: `degrees = range(1, 21)` を変更
- 評価指標の追加や保存物の拡張は `train.py` を参照