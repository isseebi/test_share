import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simulate_flexible_control(config):
    """
    物理パラメータと制御パラメータの不一致をシミュレーション。
    制御パラメータがNoneの場合は、物理パラメータから自動算出する。
    """
    # --- 1. 実システム (Plant) の特性計算 ---
    m, c, k = config['m_real'], config['c_real'], config['k_real']
    wn_real = np.sqrt(k / m)
    zeta_real = c / (2 * np.sqrt(m * k))
    
    # --- 2. 制御パラメータの決定 (柔軟なロジック) ---
    # 指定があればそれを使い、なければ実システムと同じ値（理想状態）にする
    wn_ctrl = config.get('wn_ctrl') if config.get('wn_ctrl') is not None else wn_real
    zeta_ctrl = config.get('zeta_ctrl') if config.get('zeta_ctrl') is not None else zeta_real
    
    # シミュレーション用設定
    target = config['target_pos']
    t_end = config['t_end']
    tau_ref = config['tau_ref']
    t = np.linspace(0, t_end, 2000)
    dt = t[1] - t[0]

    # 実システムの伝達関数
    sys_real = signal.TransferFunction([wn_real**2], [1, 2 * zeta_real * wn_real, wn_real**2])

    # --- 3. 制御指令の生成 (wn_ctrl, zeta_ctrl を使用) ---
    # a) ZV Shaper
    wd_c = wn_ctrl * np.sqrt(1 - zeta_ctrl**2) if zeta_ctrl < 1 else 0
    if 0 < zeta_ctrl < 1:
        K = np.exp(-zeta_ctrl * np.pi / np.sqrt(1 - zeta_ctrl**2))
        t_shaper = np.pi / wd_c
        A1, A2 = 1 / (1 + K), K / (1 + K)
        u_shaping = A1 * target * (t >= 0) + A2 * target * (t >= t_shaper)
    else:
        u_shaping = np.ones_like(t) * target

    # b) Inverse Model
    sys_ref = signal.TransferFunction([1], [tau_ref**2, 2*tau_ref, 1])
    _, y_ref, _ = signal.lsim(sys_ref, U=(np.ones_like(t) * target), T=t)
    dy_ref = np.gradient(y_ref, dt)
    d2y_ref = np.gradient(dy_ref, dt)
    u_inv = (1/wn_ctrl**2) * d2y_ref + (2*zeta_ctrl/wn_ctrl) * dy_ref + y_ref

    # --- 4. 応答の計算 ---
    _, y_step, _ = signal.lsim(sys_real, U=(np.ones_like(t) * target), T=t)
    _, y_shaping, _ = signal.lsim(sys_real, U=u_shaping, T=t)
    _, y_inv_model, _ = signal.lsim(sys_real, U=u_inv, T=t)

    # --- 5. 比較グラフ ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    
    status = "IDEAL (Auto-calc)" if (config.get('wn_ctrl') is None) else "MANUAL (Mismatch test)"
    title_text = (f"Vibration Control [{status}]\n"
                  f"Plant: wn={wn_real:.2f}, zeta={zeta_real:.3f} | "
                  f"Ctrl: wn={wn_ctrl:.2f}, zeta={zeta_ctrl:.3f}")
    
    ax1.plot(t, y_step, label='Uncompensated Step', alpha=0.5, linestyle='--')
    ax1.plot(t, y_shaping, label='Input Shaping (ZV)', linewidth=2)
    ax1.plot(t, y_inv_model, label='Inverse Model', linewidth=2)
    ax1.axhline(y=target, color='r', linestyle=':', alpha=0.3)
    ax1.set_ylabel('Displacement [m]')
    ax1.set_title(title_text)
    ax1.legend(loc='lower right')
    ax1.grid(True)

    ax2.plot(t, u_shaping, label='Shaping Input', linewidth=1.5)
    ax2.plot(t, u_inv, label='InvModel Input', linewidth=1.5)
    ax2.set_ylabel('Control Input')
    ax2.set_xlabel('Time [s]')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# パラメータ設定：None にすると自動計算されます
# ==========================================
params = {
    # 実際の物理パラメータ
    'm_real': 2.0,       # 質量 [kg]
    'c_real': 1.0,       # 粘性係数 [N・s/m]
    'k_real': 400.0,     # ばね定数 [N/m]
    
    # 制御用パラメータ（実験用）
    # None に設定すると、上の m, c, k から計算された「理想値」が使われます。
    # 数値をいれると、その値で制御指令を作成します（パラメータ不一致のテスト）。
    'wn_ctrl': None,     # 例: 15.0
    'zeta_ctrl': None,   # 例: 0.05
    # 'wn_ctrl': 20.0,     # 想定固有角周波数 (実測 wn は約15.8)
    # 'zeta_ctrl': 0.1,    # 想定減衰比 (実測 zeta は約0.158)

    'target_pos': 1.0,
    't_end': 1.5,
    'tau_ref': 0.04
}

simulate_flexible_control(params)