import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. パラメータ設定 (ここを自由に書き換えてください)
# ==========================================

# --- 設備の物理パラメータ (二次遅れ系の構成要素) ---
m = 1.0     # 質量 [kg]  (重さ)
k = 100.0   # ばね定数 [N/m]  (剛性・硬さ)
c = 0.5     # 減衰係数 [N・s/m] (ブレーキ・粘り気)

# --- 外力（モータの振動など）のパラメータ ---
F0 = 1.0    # 外力の振幅 [N]
f_ext = 1.6 # 外力の周波数 [Hz] (★ここが重要！)
            # 初期値は1.6Hz付近。後で計算される固有振動数fnに近づけてみてください。

# --- シミュレーション設定 ---
t_end = 20.0        # シミュレーション時間 [s]
dt = 0.01          # タイムステップ [s]
initial_state = [0.0, 0.0] # 初期状態 [変位x0, 速度v0]

# ==========================================
# 2. 計算とシミュレーションの実行
# ==========================================

# 1) パラメータから固有振動数と減衰比を計算
omega_n = np.sqrt(k / m)      # 固有角振動数 [rad/s]
fn = omega_n / (2 * np.pi)   # 固有振動数 [Hz]
zeta = c / (2 * np.sqrt(m * k)) # 減衰比 [-]

print("-" * 30)
print(f"設備の設計パラメータ:")
print(f"  固有振動数 fn = {fn:.3f} Hz")
print(f"  減衰比 zeta = {zeta:.4f}")
print("-" * 30)
print(f"入力（外力）パラメータ:")
print(f"  振動数 f_ext = {f_ext:.3f} Hz")
print(f"  (fn との比: {f_ext/fn:.2f})")
print("-" * 30)

# 2) 運動方程式を微分方程式系として定義
# mx'' + cx' + kx = F0 * sin(2*pi*f_ext*t)
# y1 = x, y2 = x' と置くと:
# y1' = y2
# y2' = (F0*sin - c*y2 - k*y1) / m
def equation_of_motion(t, y, m, k, c, F0, f_ext):
    x, v = y
    
    # 外力（正弦波）
    omega_ext = 2 * np.pi * f_ext
    force = F0 * np.sin(omega_ext * t)
    
    # 加速度 a = (F - damping - spring) / m
    accel = (force - c*v - k*x) / m
    
    return [v, accel]

# 3) 時間軸を作成
t_span = (0.0, t_end)
t_eval = np.arange(0.0, t_end, dt)

# 4) 微分方程式を解く (solve_ivpを使用)
solution = solve_ivp(
    equation_of_motion,
    t_span,
    initial_state,
    args=(m, k, c, F0, f_ext),
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-8
)

# ==========================================
# 3. 結果のプロット
# ==========================================

t = solution.t
x = solution.y[0]
force_ext = F0 * np.sin(2 * np.pi * f_ext * t)

# グラフの作成
plt.figure(figsize=(12, 8))

# 上段: 変位 (Displacement)
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Displacement x(t)', color='blue')
# タイトルに LaTeX 形式の数式を使用
plt.title(f'Mechanical Resonance Simulation ($f_n$={fn:.2f}Hz, $f_{{ext}}$={f_ext:.2f}Hz, $\zeta$={zeta:.3f})')
plt.ylabel('Displacement [m]')
plt.grid(True)
plt.legend(loc='upper right')

# 下段: 外力 (External Force)
plt.subplot(2, 1, 2)
plt.plot(t, force_ext, label='External Force F(t)', color='orange', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()

# --- グラフの保存 ---
# 引数にファイル名を指定します。拡張子（.png, .pdf, .svg, .jpg など）により形式が自動判別されます。
# dpi: 解像度（300以上を推奨）。
# bbox_inches='tight': ラベルが画像からはみ出すのを防ぎます。
plt.savefig('resonance_simulation.png', dpi=300, bbox_inches='tight')

# 画面に表示もしたい場合は savefig の後に呼び出します
plt.show()