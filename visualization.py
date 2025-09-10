import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import os

def plot_degree_vs_mae(degrees: List[int], train_maes: List[float], val_maes: List[float], save_path: str):
    """
    次数とMAEの関係をプロットする関数
    
    Args:
        degrees (List[int]): 次数のリスト
        train_maes (List[float]): 訓練データのMAE
        val_maes (List[float]): 検証データのMAE
        save_path (str): プロット保存先のパス
    """
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_maes, 'b-o', label='Training MAE')
    plt.plot(degrees, val_maes, 'r-o', label='Validation MAE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Absolute Error')
    plt.title('Polynomial Degree vs MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, degree: int, save_path: str):
    """
    予測値と実測値の散布図をプロットする関数
    
    Args:
        y_true (np.ndarray): 実測値
        y_pred (np.ndarray): 予測値
        degree (int): 多項式の次数
        save_path (str): プロット保存先のパス
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Distance')
    plt.ylabel('Predicted Distance')
    plt.title(f'Actual vs Predicted Distance (Degree {degree})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, degree: int, save_path: str):
    """
    残差プロットを生成する関数
    
    Args:
        y_true (np.ndarray): 実測値
        y_pred (np.ndarray): 予測値
        degree (int): 多項式の次数
        save_path (str): プロット保存先のパス
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Distance')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot (Degree {degree})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray, 
                       val_scores: np.ndarray, degree: int, save_path: str):
    """
    学習曲線をプロットする関数
    
    Args:
        train_sizes (np.ndarray): 訓練データサイズ
        train_scores (np.ndarray): 訓練スコア
        val_scores (np.ndarray): 検証スコア
        degree (int): 多項式の次数
        save_path (str): プロット保存先のパス
    """
    plt.figure(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score (MAE)')
    plt.title(f'Learning Curve (Degree {degree})')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()