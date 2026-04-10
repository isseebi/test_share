import numpy as np
import matplotlib.pyplot as plt

def simulate_slider_nozzle_control(wn, zeta, Kp, Kd, target_pos, shaper_wn, shaper_zeta, time_max=3.0, dt=0.001):
    """
    スライダのPD制御と、それに伴うノズルの慣性振動シミュレーション
    (ZVインプットシェーピング機能付き)
    """
    t = np.arange(0, time_max, dt)
    n_steps = len(t)
    
    # 物理パラメータ
    M = 1.0  # スライダの質量
    m = 0.1  # ノズル（先端負荷）の質量
    
    # ノズルとスライダを繋ぐ剛性と減衰
    k = m * wn**2
    c = 2 * m * zeta * wn
    
    # ----------------------------------------------------
    # ZV (Zero Vibration) インプットシェーパーの設計
    # ----------------------------------------------------
    use_shaping = shaper_wn > 0
    if use_shaping:
        # 減衰固有振動数 wd
        wd = shaper_wn * np.sqrt(1 - shaper_zeta**2)
        # インパルスの振幅比を決定する係数 K
        K_val = np.exp(-shaper_zeta * np.pi / np.sqrt(1 - shaper_zeta**2))
        
        A1 = 1 / (1 + K_val)     # 1回目のステップの振幅割合
        A2 = K_val / (1 + K_val) # 2回目のステップの振幅割合
        t2 = np.pi / wd          # 2回目のステップを入れるタイミング（半周期後）
    else:
        A1, A2, t2 = 1.0, 0.0, 0.0

    # 状態変数の初期化
    xs = np.zeros(n_steps) # スライダ位置
    vs = np.zeros(n_steps) # スライダ速度
    xn = np.zeros(n_steps) # ノズル位置
    vn = np.zeros(n_steps) # ノズル速度
    u = np.zeros(n_steps)  # モーターの制御入力(推力)
    
    target_shaped = np.zeros(n_steps) # シェーピングされた目標軌道
    
    for i in range(1, n_steps):
        current_time = i * dt
        
        # インプットシェーピングによる目標値の階段状変化
        if use_shaping:
            if current_time < t2:
                curr_target = target_pos * A1
            else:
                curr_target = target_pos * (A1 + A2) # 最終的に target_pos に到達
        else:
            curr_target = target_pos
            
        target_shaped[i] = curr_target

        # 現在の状態
        xs_curr = xs[i-1]
        vs_curr = vs[i-1]
        xn_curr = xn[i-1]
        vn_curr = vn[i-1]
        
        # PD制御器による推力計算 (シェーピングされた目標位置を使用)
        F_ctrl = Kp * (curr_target - xs_curr) - Kd * vs_curr
        u[i] = F_ctrl
        
        # スライダとノズル間に働く力 (バネ・ダンパによる相互作用)
        F_interaction = k * (xs_curr - xn_curr) + c * (vs_curr - vn_curr)
        
        # 運動方程式に基づく加速度の計算
        a_s = (F_ctrl - F_interaction) / M
        a_n = F_interaction / m
        
        # 状態の更新 (オイラー法)
        vs[i] = vs_curr + a_s * dt
        xs[i] = xs_curr + vs[i] * dt
        
        vn[i] = vn_curr + a_n * dt
        xn[i] = xn_curr + vn[i] * dt
        
    target_shaped[0] = target_pos * A1 if use_shaping else target_pos

    return t, xs, xn, u, target_shaped

# プラント（実際の物理モデル）のパラメータ
wn_real = 80.0   
zeta_real = 0.05 

# =========================================================
# 【ユーザー設定エリア】色々変更して挙動を比較してください
# =========================================================
Kp_setting = 80.0
Kd_setting = 10.0
target = 1.0

# --- インプットシェーパー（ZVシェーパー）の設定 ---
# 制御側が想定する対象の「固有振動数」と「減衰率」を指定します。

# 【正解のパラメータ】(実際の物理モデルと完全に一致させた場合)
# 以下のコメントアウトを外し、テスト用パラメータを上書きすると
# 理論上、ノズルの残留振動が最も綺麗に打ち消されます。
# shaper_wn_setting = 30.0
# shaper_zeta_setting = 0.05

# 【テスト用パラメータ】(想定がズレている場合をテスト)
# わざと実際の値(30.0, 0.05)からズラすことで、ロバスト性を確認できます。
shaper_wn_setting = 80.0
shaper_zeta_setting = 0.01
# =========================================================

# 1. シェーピングなし（比較用：shaper_wn=0で無効化）
t, xs_base, xn_base, u_base, target_base = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=0, shaper_zeta=0)

# 2. シェーピングあり
t, xs_shape, xn_shape, u_shape, target_shape = simulate_slider_nozzle_control(
    wn_real, zeta_real, Kp_setting, Kd_setting, target, 
    shaper_wn=shaper_wn_setting, shaper_zeta=shaper_zeta_setting)

# -----------------
# 結果のプロット
# -----------------
plt.figure(figsize=(10, 10))

# 1. 絶対位置のグラフ（シェーピングあり/なしのノズル位置比較）
plt.subplot(3, 1, 1)
plt.plot(t, target_shape, label='Shaped Target (Input)', color='green', linestyle=':', linewidth=2)
plt.plot(t, xn_base, label='Nozzle (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, xn_shape, label='Nozzle (With Shaping)', color='red')
plt.axhline(target, color='black', linestyle='--', alpha=0.5, label='Final Target')
plt.title('Nozzle Position: No Shaping vs With Shaping')
plt.ylabel('Absolute Position')
plt.legend()
plt.grid(True)

# 2. ノズルの揺れ（スライダからの相対変位）
plt.subplot(3, 1, 2)
deflection_base = xn_base - xs_base
deflection_shape = xn_shape - xs_shape
plt.plot(t, deflection_base, label='Deflection (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, deflection_shape, label='Deflection (With Shaping)', color='orange')
plt.ylabel('Deflection (xn - xs)')
plt.legend()
plt.grid(True)

# 3. 制御入力のグラフ
plt.subplot(3, 1, 3)
plt.plot(t, u_base, label='Control Effort (No Shaping)', color='gray', alpha=0.5)
plt.plot(t, u_shape, label='Control Effort (With Shaping)', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Force')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()