import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Dict, List

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    データの読み込みと前処理を行う関数
    
    Args:
        file_path (str): 入力データのパス
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 特徴量とターゲット
    """
    # データ読み込み
    df = pd.read_csv(file_path)
    
    # 特徴量とターゲットの分離
    X = df[['under_y', 'theta']].values
    y = df['distance'].values
    
    return X, y

def create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    多項式特徴量を生成する関数
    
    Args:
        X (np.ndarray): 入力特徴量
        degree (int): 多項式の次数
    
    Returns:
        np.ndarray: 生成された多項式特徴量
    """
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    特徴量のスケーリングを行う関数
    
    Args:
        X (np.ndarray): 入力特徴量
    
    Returns:
        Tuple[np.ndarray, StandardScaler]: スケーリングされた特徴量とスケーラー
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def evaluate_model(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
    """
    モデルの評価を行う関数
    
    Args:
        model: 評価するモデル
        X (np.ndarray): 特徴量
        y (np.ndarray): ターゲット
        n_splits (int): 交差検証の分割数
    
    Returns:
        Dict: 評価結果の辞書
    """
    # 交差検証によるMAEの計算
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = cross_val_score(
        model, X, y, 
        scoring='neg_mean_absolute_error',
        cv=kfold
    )
    
    # スコアを正の値に変換
    mae_scores = -mae_scores
    
    return {
        'mean_mae': mae_scores.mean(),
        'std_mae': mae_scores.std(),
        'all_scores': mae_scores
    }

def ensure_dir(directory: str):
    """
    ディレクトリが存在しない場合は作成する関数
    
    Args:
        directory (str): 作成するディレクトリのパス
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_filename(degree: int, mae: float) -> str:
    """
    モデルファイル名のフォーマットを行う関数
    
    Args:
        degree (int): 多項式の次数
        mae (float): MAEスコア
    
    Returns:
        str: フォーマットされたファイル名
    """
    return f"polynomial_degree{degree}_mae{mae:.2f}.joblib"