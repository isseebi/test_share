import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. システム定義（物理パラメータの設定） ---
# システム1: 低周波（k1=100）
m1, k1, c1 = 1.0, 100.0, 1.0
# システム2: 高周波（k2=400）
m2, k2, c2 = 1.0, 400.0, 1.0
# 全体の入力速度（パルスの総和を調整するための係数）
v_total = 2.0

# --- 2. ZVシェーパーの計算関数 ---
def get_zv_params(m, k, c):
    """
    質量m, 剛性k, 減衰cから、振動を打ち消す2つのパルスの時間と振幅を計算する
    """
    omega_n = np.sqrt(k / m)                        # 固有角振動数
    zeta = c / (2 * np.sqrt(m * k))                 # 減衰比
    omega_d = omega_n * np.sqrt(1 - zeta**2)        # 減衰固有角振動数
    
    td = np.pi / omega_d                            # 2番目のパルスを打つタイミング（半周期）
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) # 減衰による振幅の減少率
    
    a1 = 1 / (1 + K)                                # 1番目のパルスの振幅
    a2 = K / (1 + K)                                # 2番目のパルスの振幅
    return [0.0, td], [a1, a2]                      # [時間リスト], [振幅リスト]

# 各システム用のZVパルスを取得
times1, amps1 = get_zv_params(m1, k1, c1)
times2, amps2 = get_zv_params(m2, k2, c2)

# --- 3. 2つのシェーパーの畳み込み（マルチモード化） ---
# システム1の2パルス × システム2の2パルス = 合計4パルスの合成パルスを作る
combined_pulses = []
for t1, a1 in zip(times1, amps1):
    for t2, a2 in zip(times2, amps2):
        # 時間は足し算、振幅は掛け算（これが畳み込みの原理）
        combined_pulses.append((t1 + t2, a1 * a2))

# 時間順に並べ替え
combined_pulses.sort()
final_times = [p[0] for p in combined_pulses]
final_amps = [p[1] for p in combined_pulses]

# --- 4. シミュレーション用の関数定義 ---
def system_dynamics(state, t, m, k, c):
    """単振動（減衰あり）の微分方程式: mx'' + cx' + kx = 0"""
    x, v = state
    return [v, (-k * x - c * v) / m]

t_span = np.linspace(0, 4, 2000) # シミュレーション時間（0~4秒）

def simulate_multi_shaping(times, amps, m, k, c):
    """
    複数のパルス入力に対するシステムの応答を計算する
    """
    res_combined = np.zeros((len(t_span), 2)) # 結果格納用
    current_state = np.array([0.0, 0.0])      # [初期変位, 初期速度]
    
    for i in range(len(times)):
        t_start = times[i]
        # 次のパルスまでの間、あるいは最後までを計算区間とする
        t_end = times[i+1] if i+1 < len(times) else t_span[-1]
        
        # パルス入力：指定時刻に速度（v）を瞬間的に変化させる（インパルス入力の近似）
        current_state[1] += amps[i] * v_total
        
        # 該当する時間インデックスを抽出
        idx_start = np.searchsorted(t_span, t_start)
        idx_end = np.searchsorted(t_span, t_end)
        
        if idx_start < idx_end:
            # 物理エンジンのように、その区間の微分方程式を解く
            t_seg = t_span[idx_start:idx_end+1] - t_span[idx_start]
            res_seg = odeint(system_dynamics, current_state, t_seg, args=(m, k, c))
            
            # 結果をメインの配列に格納
            res_combined[idx_start:idx_end+1] = res_seg
            # 次の区間の開始状態として、現在の最終状態を保存
            current_state = res_seg[-1].copy()
            
    return res_combined

# --- 5. 比較用データの計算 ---

# A. 制御なし（時刻0で一度だけ強く叩く）
res_unshaped_sys1 = simulate_multi_shaping([0.0], [1.0], m1, k1, c1)
res_unshaped_sys2 = simulate_multi_shaping([0.0], [1.0], m2, k2, c2)
res_unshaped_mixed = res_unshaped_sys1 + res_unshaped_sys2 # 2つの波形の重ね合わせ

# B. 制御あり（計算した4つのパルスで分散して叩く）
res_shaped_sys1 = simulate_multi_shaping(final_times, final_amps, m1, k1, c1)
res_shaped_sys2 = simulate_multi_shaping(final_times, final_amps, m2, k2, c2)
res_shaped_mixed = res_shaped_sys1 + res_shaped_sys2

# --- 6. 可視化（グラフ描画） ---
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

# グラフ1: 入力信号の比較（いつ、どの強さで叩いているか）
axes[0].stem(final_times, final_amps, linefmt='r-', markerfmt='ro', label='Multi-mode Shaper Pulses')
axes[0].stem([0.0], [1.0], linefmt='k--', markerfmt='kx', label='Unshaped Pulse (Single)')
axes[0].set_title('Input Pulse Comparison')
axes[0].set_ylabel('Amplitude')
axes[0].legend()

# グラフ2: システムごとの個別波形
# 点線が制御なし（揺れ続ける）、実線が制御あり（パルスが打ち終わるとピタッと止まる）
axes[1].plot(t_span, res_unshaped_sys1[:, 0], 'b:', alpha=0.4, label='Unshaped Wave 1 (Low Freq)')
axes[1].plot(t_span, res_shaped_sys1[:, 0], 'b-', label='Shaped Wave 1')
axes[1].plot(t_span, res_unshaped_sys2[:, 0], 'g:', alpha=0.4, label='Unshaped Wave 2 (High Freq)')
axes[1].plot(t_span, res_shaped_sys2[:, 0], 'g-', label='Shaped Wave 2')
axes[1].set_title('Individual Wave Comparison (Unshaped vs Shaped)')
axes[1].set_ylabel('Displacement [m]')
axes[1].legend(loc='upper right', fontsize='small')

# グラフ3: 2つを合計した全体の応答
# 制御あり（赤実線）が、最後のパルス以降に振動がほぼゼロになっていることがゴール
axes[2].plot(t_span, res_unshaped_mixed[:, 0], 'k:', alpha=0.3, label='Unshaped Mixed (Heavy Vibration)')
axes[2].plot(t_span, res_shaped_mixed[:, 0], 'r-', linewidth=2, label='Shaped Mixed (Suppressed)')
axes[2].set_title('Total System Response (The Goal of Control)')
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Displacement [m]')
axes[2].legend()

# 仕上げ
for ax in axes: ax.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()