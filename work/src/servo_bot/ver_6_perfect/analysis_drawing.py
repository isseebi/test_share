import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

def calculate_stats(history: Dict[str, np.ndarray], target_val: float) -> Dict[str, Optional[float]]:
    """
    シミュレーション結果から統計情報（オーバーシュート、制定時間）を計算する。
    """
    m_pos = history["motor_pos"]
    time = history["time"]
    
    # オーバーシュート
    max_pos = np.max(m_pos)
    overshoot = max(0, max_pos - target_val)
    
    # 制定時間の計算 (±2%)
    threshold = abs(target_val) * 0.02
    settled_idx = -1
    # 後ろから見て、最初に閾値を超えた場所を探す
    for i in range(len(m_pos)-1, -1, -1):
        if abs(m_pos[i] - target_val) > threshold:
            settled_idx = i
            break
    
    if settled_idx == -1:
        settling_time = 0.0
    elif settled_idx == len(m_pos) - 1:
        settling_time = None # 未収束
    else:
        settling_time = time[settled_idx + 1]
        
    return {
        "overshoot": overshoot,
        "settling_time": settling_time,
        "max_pos": max_pos
    }

def create_servo_figure(history: Dict[str, np.ndarray], 
                        target_val: float, 
                        mode: str, 
                        use_cmd_lpf: bool,
                        prev_history: Optional[Dict[str, np.ndarray]] = None,
                        zoom: bool = False) -> plt.Figure:
    """
    サーボシミュレーションのグラフを作成する（Dark Mode相当）。
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    time = history["time"]
    
    # 目標値 (Raw)
    ax.plot(time, history["target_raw"], color='white', linestyle=':', label='Target (Raw)', alpha=0.4)
    
    # 指令LPF
    if use_cmd_lpf:
        ax.plot(time, history["target_filtered"], color='#00FF00', linestyle='--', label='Target (Filtered)', alpha=0.8)
    
    # モータ位置 (現在)
    ax.plot(time, history["motor_pos"], color='#00D4FF', linewidth=2, label='Motor Position', alpha=0.9)
    
    # モータ位置 (前回 - ゴースト表示)
    if prev_history is not None:
        ax.plot(prev_history["time"], prev_history["motor_pos"], 
                color='#00D4FF', linewidth=1.5, linestyle='--', label='Prev Position', alpha=0.15)
    
    title = f"Overview: {mode}" if not zoom else "Settling View (Zoomed)"
    ax.set_title(title, fontsize=12, pad=10, color='white')
    ax.set_xlabel('Time [s]', color='#AAAAAA')
    ax.set_ylabel('Position', color='#AAAAAA')
    ax.grid(True, linestyle='--', alpha=0.2, color='#555555')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    
    # ズーム設定
    if zoom:
        stats = calculate_stats(history, target_val)
        overshoot_val = stats["overshoot"]
        margin = max(overshoot_val * 1.5, target_val * 0.02)
        ax.set_ylim(target_val - margin, target_val + margin)
    else:
        ax.legend(facecolor='#1E1E1E', edgecolor='#333333', fontsize='small')
    
    plt.tight_layout()
    return fig
