import numpy as np
import matplotlib.pyplot as plt

def simulate_slider_nozzle_control(wn, zeta, Kp, Kd, target_pos, shaper_wn, shaper_zeta, time_max=3.0, dt=0.001):
    """
    スライダのPD制御と、それに伴うノズルの慣性振動シミュレーション
    """
    t = np.arange(0, time_max, dt)
    n_steps = len(t)
    
    M = 1.0  # スライダの質量
    m = 0.1  # ノズル（先端負荷）の質量
    
    k = m * wn**2
    c = 2 * m * zeta * wn
    
    use_shaping = shaper_wn > 0
    if use_shaping:
        wd = shaper_wn * np.sqrt(1 - shaper_zeta**2)
        K_val = np.exp(-shaper_zeta * np.pi / np.sqrt(1 - shaper_zeta**2))
        A1 = 1 / (1 + K_val)
        A2 = K_val / (1 + K_val)
        t2 = np.pi / wd
    else:
        A1, A2, t2 = 1.0, 0.0, 0.0

    xs = np.zeros(n_steps)
    vs = np.zeros(n_steps)
    xn = np.zeros(n_steps)
    vn = np.zeros(n_steps)
    u = np.zeros(n_steps)
    target_shaped = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        current_time = i * dt
        if use_shaping:
            curr_target = target_pos * A1 if current_time < t2 else target_pos * (A1 + A2)
        else:
            curr_target = target_pos
        target_shaped[i] = curr_target

        xs_curr, vs_curr = xs[i-1], vs[i-1]
        xn_curr, vn_curr = xn[i-1], vn[i-1]
        
        F_ctrl = Kp * (curr_target - xs_curr) - Kd * vs_curr
        u[i] = F_ctrl
        F_interaction = k * (xs_curr - xn_curr) + c * (vs_curr - vn_curr)
        
        a_s = (F_ctrl - F_interaction) / M
        a_n = F_interaction / m
        
        vs[i] = vs_curr + a_s * dt
        xs[i] = xs_curr + vs[i] * dt
        vn[i] = vn_curr + a_n * dt
        xn[i] = xn_curr + vn[i] * dt
        
    target_shaped[0] = target_pos * A1 if use_shaping else target_pos
    return t, xs, xn, u, target_shaped

# --- パラメータ設定 ---
wn_real = 80.0   
zeta_real = 0.05 

Kp_setting = 80.0
Kd_setting = 5.0
target = 1.0
dt = 0.001

# テスト用パラメータ
shaper_wn_setting = 80.0
shaper_zeta_setting = 0.05

# --- シミュレーション実行 ---
t, xs_base, xn_base, u_base, _ = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 0, 0, dt=dt)

t, xs_shape, xn_shape, u_shape, target_shape = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn_setting, shaper_zeta_setting, dt=dt)

# --- FFT解析処理 ---
def compute_fft(signal, dt):
    n = len(signal)
    # 信号の平均を引く（DCオフセット除去：グラフを見やすくするため）
    signal_detrended = signal - np.mean(signal)
    fft_val = np.fft.fft(signal_detrended)
    fft_freq = np.fft.fftfreq(n, dt)
    
    # 正の周波数領域のみ抽出
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    # 振幅を正規化
    magnitude = np.abs(fft_val[pos_mask]) * (2.0 / n)
    return freqs, magnitude

# 各信号のFFT実行（特にノズル変位の振動成分を解析）
deflection_base = xn_base - xs_base
deflection_shape = xn_shape - xs_shape

f_base, mag_base = compute_fft(deflection_base, dt)
f_shape, mag_shape = compute_fft(deflection_shape, dt)

# --- 結果のプロット ---
fig, axs = plt.subplots(4, 1, figsize=(10, 14))

# 1. ノズル絶対位置
axs[0].plot(t, xn_base, label='Nozzle (No Shaping)', color='gray', alpha=0.5)
axs[0].plot(t, xn_shape, label='Nozzle (With Shaping)', color='red')
axs[0].axhline(target, color='black', linestyle='--', alpha=0.5)
axs[0].set_title('Nozzle Position')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

# 2. ノズルの揺れ（相対変位）
axs[1].plot(t, deflection_base, label='Deflection (No Shaping)', color='gray', alpha=0.5)
axs[1].plot(t, deflection_shape, label='Deflection (With Shaping)', color='orange')
axs[1].set_title('Nozzle Deflection (xn - xs)')
axs[1].set_ylabel('Deflection')
axs[1].legend()
axs[1].grid(True)

# 3. 周波数スペクトル (FFT)
axs[2].plot(f_base, mag_base, label='Spectrum (No Shaping)', color='gray', alpha=0.5)
axs[2].plot(f_shape, mag_shape, label='Spectrum (With Shaping)', color='blue')
# 理論上の固有振動数を表示 (Hz = rad/s / 2pi)
theory_hz = wn_real / (2 * np.pi)
axs[2].axvline(theory_hz, color='green', linestyle=':', label=f'Target Freq ({theory_hz:.1f}Hz)')
axs[2].set_xlim(0, 100) # 0〜100Hz付近を拡大
axs[2].set_title('Frequency Spectrum (FFT of Deflection)')
axs[2].set_xlabel('Frequency [Hz]')
axs[2].set_ylabel('Magnitude')
axs[2].legend()
axs[2].grid(True)

# 4. 制御入力
axs[3].plot(t, u_base, color='gray', alpha=0.5, label='Control (No Shaping)')
axs[3].plot(t, u_shape, color='purple', label='Control (With Shaping)')
axs[3].set_title('Control Effort (Force)')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Force')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()