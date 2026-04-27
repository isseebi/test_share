import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any
from analysis_drawing import calculate_stats, create_servo_figure

# 日本語文字化け対策
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

class AnalysisService:
    """統計計算、画像生成のラップ"""
    
    @staticmethod
    def get_stats(history: Dict[str, np.ndarray], target_val: float) -> Dict[str, Optional[float]]:
        return calculate_stats(history, target_val)

    @staticmethod
    def create_figure(history: Dict[str, np.ndarray], 
                      target_val: float, 
                      mode: str, 
                      use_cmd_lpf: bool,
                      prev_history: Optional[Dict[str, np.ndarray]] = None,
                      zoom: bool = False) -> plt.Figure:
        return create_servo_figure(history, target_val, mode, use_cmd_lpf, prev_history, zoom)
