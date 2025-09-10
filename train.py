import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split # 追加
import joblib
import os
import csv # 追加
from datetime import datetime
from utils import (
    load_data,
    create_polynomial_features,
    scale_features,
    ensure_dir,
    format_filename
)
from visualization import (
    plot_degree_vs_mae,
    plot_predictions_vs_actual,
    plot_residuals
)

def train_polynomial_regression(X: np.ndarray, y: np.ndarray, degree: int) -> tuple:
    """
    指定された次数で多項式回帰モデルを学習する関数
    
    Args:
        X (np.ndarray): 入力特徴量
        y (np.ndarray): ターゲット
        degree (int): 多項式の次数
    
    Returns:
        tuple: (学習済みモデル, スケーラー)
    """
    # 多項式特徴量の生成
    X_poly = create_polynomial_features(X, degree)
    
    # 特徴量のスケーリング
    X_scaled, scaler = scale_features(X_poly)
    
    # モデルの学習
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

def main():
    # 現在時刻を取得してタイムスタンプディレクトリを作成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join('models', timestamp)
    result_dir = os.path.join('results', timestamp)
    
    # ディレクトリの作成
    ensure_dir(model_dir)
    ensure_dir(result_dir)
    
    # 学習データと検証データの読み込み
    train_path = "data/predictions_alloutput_0716_10divid_separate_128,128.csv"
    val_path = "data/計測データ0715_processed.csv"

    # 追加: 学習データ比率（0.5なら7割のうち半分のみ学習に使用）
    train_sub_ratio = 1.0  # 0.0～1.0で指定

    if train_path == val_path:
        print(f"学習データと検証データに同じファイル ({train_path}) が指定されました。データを7:3に分割します。")
        X_all, y_all = load_data(train_path)
        X_train_full, X_val, y_train_full, y_val = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
        # 7割のうち指定比率だけを学習に使用
        n_train = int(len(X_train_full) * train_sub_ratio)
        X_train = X_train_full[:n_train]
        y_train = y_train_full[:n_train]
        print(f"学習データ数（使用分）: {len(X_train)} / {len(X_train_full)}")
        print(f"検証データ数: {len(X_val)}")
    else:
        print(f"学習データ: {train_path}")
        print(f"検証データ: {val_path}")
        X_train_full, y_train_full = load_data(train_path)
        X_val, y_val = load_data(val_path)
        n_train = int(len(X_train_full) * train_sub_ratio)
        X_train = X_train_full[:n_train]
        y_train = y_train_full[:n_train]
        print(f"学習データ数（使用分）: {len(X_train)} / {len(X_train_full)}")
        print(f"検証データ数: {len(X_val)}")
    
    # 1次から20次までの多項式回帰を実行
    degrees = range(1, 21)
    results = []
    best_mae = float('inf')
    best_degree = None
    
    for degree in degrees:
        print(f"\n次数 {degree} の多項式回帰を実行中...")
        
        # モデルの学習
        model, scaler = train_polynomial_regression(X_train, y_train, degree)

        # 訓練データでの評価
        X_train_poly = create_polynomial_features(X_train, degree)
        X_train_scaled = scaler.transform(X_train_poly)
        y_train_pred = model.predict(X_train_scaled)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))

        # 検証データでの評価
        X_val_poly = create_polynomial_features(X_val, degree)
        X_val_scaled = scaler.transform(X_val_poly)
        val_predictions = model.predict(X_val_scaled)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(np.mean((y_val - val_predictions) ** 2))

        print(f"訓練データ - MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        print(f"検証データ - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
        
        results.append({
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        })
        
        # 予測値の取得（プロット用）
        X_train_poly = create_polynomial_features(X_train, degree)
        X_train_scaled, _ = scale_features(X_train_poly)
        y_train_pred = model.predict(X_train_scaled)
        
        # 可視化
        plot_predictions_vs_actual(
            y_train, y_train_pred, degree,
            os.path.join(result_dir, f'predictions_degree_{degree}.png')
        )
        plot_residuals(
            y_train, y_train_pred, degree,
            os.path.join(result_dir, f'residuals_degree_{degree}.png')
        )
        # モデルの保存
        model_filename = format_filename(degree, val_mae)
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'degree': degree,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        }, os.path.join(model_dir, model_filename))
        
        # 最良モデルの更新
        if val_mae < best_mae:
            best_mae = val_mae
            best_degree = degree
        
    
    # 次数とMAEの関係をプロット
    plot_degree_vs_mae(
        list(degrees),
        [result['train_mae'] for result in results],
        [result['val_mae'] for result in results],
        os.path.join(result_dir, 'degree_vs_mae.png')
    )

    # MAE結果をCSVファイルに保存
    mae_csv_path = os.path.join(result_dir, 'mae_results.csv')
    with open(mae_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['degree', 'val_mae'] # train_mae を削除
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, degree in enumerate(degrees):
            writer.writerow({
                'degree': degree,
                'val_mae': results[i]['val_mae'] # train_mae の書き込みを削除
            })
    print(f"\nMAE結果を {mae_csv_path} に保存しました。")
    
    print(f"\n学習完了")
    print(f"最良の次数: {best_degree}")
    print(f"最良の検証データMAE: {best_mae:.2f}")

    # 検証データでの予測vs実測値プロットの生成（最良モデル）
    best_model_path = None
    for filename in os.listdir(model_dir):
        if f'polynomial_degree{best_degree}' in filename:
            best_model_path = os.path.join(model_dir, filename)
            break
    
    if best_model_path:
        best_model_data = joblib.load(best_model_path)
        best_model = best_model_data['model']
        best_scaler = best_model_data['scaler']
        
        # 検証データでの予測
        X_val_poly = create_polynomial_features(X_val, best_degree)
        X_val_scaled = best_scaler.transform(X_val_poly)
        val_predictions = best_model.predict(X_val_scaled)
        
        # 検証データでの予測vs実測値プロット
        plot_predictions_vs_actual(
            y_val, val_predictions, best_degree,
            os.path.join(result_dir, 'validation_predictions_best_model.png')
        )
        plot_residuals(
            y_val, val_predictions, best_degree,
            os.path.join(result_dir, 'validation_residuals_best_model.png')
        )

if __name__ == "__main__":
    main()