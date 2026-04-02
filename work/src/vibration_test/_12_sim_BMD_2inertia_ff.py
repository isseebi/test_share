import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simulate_2inertia_zvdd(config):
    """
    ボールねじ(m1)と手先(m2)の2慣性系モデル。
    ZVDD(4段階)インプットシェーピングと逆モデル制御の比較。
    """
    # --- 1. 実システム (Plant) のパラメータ定義 ---
    m1, m2 = config['m1'], config['m2']
    k12, c12 = config['k12'], config['c12'] # 結合部の剛性と減衰
    c1, c2 = config['c1'], config['c2']     # 各部の地面に対する粘性摩擦
    Kp, Kd = config['Kp'], config['Kd']     # ボールねじ側のPD制御ゲイン

    # 状態空間モデルの構築
    # 状態変数 X = [x1(ボールねじ位置), v1(速度), x2(手先位置), v2(速度)]
    # 運動方程式:
    # m1*v1_dot = Kp(u - x1) - Kd*v1 - c1*v1 - k12(x1 - x2) - c12(v1 - v2)
    # m2*v2_dot = -c2*v2 + k12(x1 - x2) + c12(v1 - v2)
    
    A = np.array([
        [0, 1, 0, 0],
        [-(Kp + k12)/m1, -(Kd + c1 + c12)/m1, k12/m1, c12/m1],
        [0, 0, 0, 1],
        [k12/m2, c12/m2, -k12/m2, -(c2 + c12)/m2]
    ])
    B = np.array([[0], [Kp/m1], [0], [0]])
    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # 出力: x1, x2
    D = np.array([[0], [0]])

    sys_real = signal.StateSpace(A, B, C, D)

    # --- 2. 制御用共振パラメータの算出 ---
    # システムの固有値から、支配的な（減衰が最も悪い）共振周波数を特定
    eigenvalues = np.linalg.eigvals(A)
    complex_poles = [p for p in eigenvalues if np.imag(p) > 1e-3]
    
    if complex_poles:
        res_pole = min(complex_poles, key=lambda p: -p.real / np.abs(p))
        wn_auto = np.abs(res_pole)
        zeta_auto = -res_pole.real / wn_auto
    else:
        wn_auto, zeta_auto = 10.0, 0.1

    wn_ctrl = config.get('wn_ctrl') or wn_auto
    zeta_ctrl = config.get('zeta_ctrl') or zeta_auto

    # --- 3. 制御指令の生成 ---
    target = config['target_pos']
    t_end = config['t_end']
    t = np.linspace(0, t_end, 4000)
    dt = t[1] - t[0]

    # a) ZVDD Shaper (4段階印加)
    wd = wn_ctrl * np.sqrt(1 - zeta_ctrl**2) if zeta_ctrl < 1 else 0
    if 0 < zeta_ctrl < 1:
        K = np.exp(-zeta_ctrl * np.pi / np.sqrt(1 - zeta_ctrl**2))
        dT = np.pi / wd # 半周期
        denom = (1 + K)**3
        # 4つのインパルス強度
        A1, A2, A3, A4 = 1/denom, 3*K/denom, 3*K**2/denom, K**3/denom
        
        u_shaping = (A1 * target * (t >= 0) + 
                     A2 * target * (t >= dT) + 
                     A3 * target * (t >= 2*dT) + 
                     A4 * target * (t >= 3*dT))
    else:
        u_shaping = np.ones_like(t) * target

    # b) Inverse Model (手先の揺れを考慮した参照モデル)
    tau_ref = config['tau_ref']
    sys_ref = signal.TransferFunction([1], [tau_ref**2, 2*tau_ref, 1])
    _, y_ref, _ = signal.lsim(sys_ref, U=(np.ones_like(t) * target), T=t)
    dy_ref = np.gradient(y_ref, dt)
    d2y_ref = np.gradient(dy_ref, dt)
    u_inv = (1/wn_ctrl**2) * d2y_ref + (2*zeta_ctrl/wn_ctrl) * dy_ref + y_ref

    # --- 4. 応答の計算 ---
    _, y_step, _ = signal.lsim(sys_real, U=(np.ones_like(t) * target), T=t)
    _, y_shaping, _ = signal.lsim(sys_real, U=u_shaping, T=t)
    _, y_inv, _ = signal.lsim(sys_real, U=u_inv, T=t)

    # --- 5. 比較グラフ ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # グラフ1: 手先 (x2) の応答
    axes[0].plot(t, y_step[:, 1], label='Uncompensated Step', alpha=0.4, linestyle='--')
    axes[0].plot(t, y_shaping[:, 1], label='ZVDD Shaping (4-step)', linewidth=2)
    axes[0].plot(t, y_inv[:, 1], label='Inverse Model', linewidth=2)
    axes[0].set_ylabel('End-Effector (x2) [m]')
    axes[0].set_title(f"2-Inertia Control: Resonance {wn_auto:.2f} rad/s, zeta {zeta_auto:.3f}")
    axes[0].legend(loc='lower right')
    axes[0].grid(True)

    # グラフ2: ボールねじ (x1) の応答
    axes[1].plot(t, y_step[:, 0], alpha=0.4, linestyle='--')
    axes[1].plot(t, y_shaping[:, 0], linewidth=2)
    axes[1].plot(t, y_inv[:, 0], linewidth=2)
    axes[1].set_ylabel('Ball-Screw (x1) [m]')
    axes[1].grid(True)

    # グラフ3: 制御入力 (指令値)
    axes[2].plot(t, u_shaping, label='ZVDD 4-step Input', color='orange')
    axes[2].plot(t, u_inv, label='InvModel Input', color='green')
    axes[2].set_ylabel('Command Input')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend(loc='lower right')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# パラメータ設定
# ==========================================
params = {
    # 物理パラメータ
    'm1': 1.0,      # ボールねじ可動部の質量 [kg]
    'm2': 0.2,      # 手先の質量 [kg]
    'k12': 400.0,   # ボールねじ-手先間の剛性 [N/m]
    'c12': 1.5,     # 結合部の減衰 [N・s/m]
    'c1': 5.0,      # ボールねじ部の摩擦抵抗 [N・s/m]
    'c2': 0.5,      # 手先部の摩擦抵抗（通常小さい） [N・s/m]
    
    # サーボ制御ゲイン
    'Kp': 8000.0,
    'Kd': 150.0,
    
    # 制御用ターゲット（Noneなら自動推定）
    'wn_ctrl': None,
    'zeta_ctrl': None,
    
    'target_pos': 1.0,
    't_end': 2.0,
    'tau_ref': 0.05
}

simulate_2inertia_zvdd(params)