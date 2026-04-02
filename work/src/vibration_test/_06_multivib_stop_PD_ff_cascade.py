import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 制御対象パラメータ (3慣性系モデル: モータ + テーブル + 架台)
# ==========================================
m1 = 1.0     # モータ側の等価質量 [kg]
m2 = 1.0     # 負荷側(テーブル)の質量 [kg]
k  = 2000.0  # ボールねじの等価剛性 [N/m]
c  = 2.0     # ボールねじの粘性減衰係数 [N/(m/s)]

m3 = 5.0     # 設備本体(架台)の質量 [kg]
k3 = 20000.0 # 架台を支えるマウントの剛性 [N/m]
c3 = 20.0    # 架台マウントの減衰係数 [N/(m/s)]

# 2つの主要な共振周波数と減衰比（近似的な非連成モデルとして計算）
omega_table = np.sqrt(k / m2)
zeta_table  = c / (2 * np.sqrt(k * m2))

omega_base = np.sqrt(k3 / m3)
zeta_base  = c3 / (2 * np.sqrt(k3 * m3))

# ==========================================
# 2. パルス入力 (2モード ZV Shaper)
# ==========================================
# テーブル用と架台用の2つのシェーパを作り、それらを「畳み込み(コンボリューション)」して
# 4つのパルスを持つマルチモードシェーパを作成します。
def get_zv_shaper(omega, zeta):
    omega_d = omega * np.sqrt(1 - zeta**2)
    Td = np.pi / omega_d
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    return np.array([1 / (1 + K), K / (1 + K)]), np.array([0, Td])

A_t, T_t = get_zv_shaper(omega_table, zeta_table)
A_b, T_b = get_zv_shaper(omega_base, zeta_base)

# 2つのシェーパを掛け合わせて4インパルスにする
impulses = []
for i in range(2):
    for j in range(2):
        impulses.append({'A': A_t[i] * A_b[j], 'T': T_t[i] + T_b[j]})

# ==========================================
# 3. 制御器ゲイン & 4. 目標軌道生成
# ==========================================
Kp = 20000.0  
Kd = 400.0    

dt = 0.001
t = np.arange(0, 1.0, dt)
N = len(t)

r = np.zeros(N)
v = np.zeros(N)
a = np.zeros(N)

for i in range(1, N):
    if 0.1 <= t[i] < 0.3:
        a[i] = 10.0
    elif 0.3 <= t[i] < 0.5:
        a[i] = -10.0
    else:
        a[i] = 0.0
    
    v[i] = v[i-1] + a[i] * dt
    r[i] = r[i-1] + v[i] * dt

# ==========================================
# 5. シェーピング軌道の作成 (4パルスの畳み込み)
# ==========================================
r_shaped = np.zeros(N)
a_shaped = np.zeros(N)

for imp in impulses:
    shift = int(imp['T'] / dt)
    for i in range(shift, N):
        r_shaped[i] += imp['A'] * r[i - shift]
        a_shaped[i] += imp['A'] * a[i - shift]

# ==========================================
# 6. シミュレーション実行関数
# ==========================================
def simulate(use_shaping=True):
    xr, vr = 0.0, 0.0 # モータ(ロータ)の「架台に対する相対」位置・速度
    xt, vt = 0.0, 0.0 # テーブルの「絶対」位置・速度
    xb, vb = 0.0, 0.0 # 架台の「絶対」位置・速度
    
    xt_log, xb_log = [], []
    
    ref_r = r_shaped if use_shaping else r
    ref_a = a_shaped if use_shaping else a
    
    for i in range(N):
        # --- (A) 制御入力 ---
        u_ff = (m1 + m2) * ref_a[i]
        
        # エンコーダは「架台上のモータの回転」を測るため、フィードバックには xr を使う
        err = ref_r[i] - xr
        u_fb = Kp * err - Kd * vr 
        u = u_ff + u_fb
        
        # --- (B) 制御対象のダイナミクス (3慣性系) ---
        # ボールねじの変位: (架台位置 + モータ相対位置) - テーブル位置
        dx = (xb + xr) - xt
        dv = (vb + vr) - vt
        F_spring = k * dx + c * dv
        
        # 運動方程式
        # 1. モータ(相対運動): 推力で進み、ボールねじの反力を受ける
        ar = (u - F_spring) / m1  
        # 2. テーブル(絶対運動): ボールねじに引っ張られて進む
        at = F_spring / m2        
        # 3. 架台(絶対運動): ボールねじの反力(-F_spring)で後ろに蹴られる
        ab = (-F_spring - k3 * xb - c3 * vb) / m3 
        
        # 積分
        vr += ar * dt
        xr += vr * dt
        vt += at * dt
        xt += vt * dt
        vb += ab * dt
        xb += vb * dt
        
        xt_log.append(xt)
        xb_log.append(xb)
        
    return np.array(xt_log), np.array(xb_log)

xt_normal, xb_normal = simulate(use_shaping=False)
xt_shaped, xb_shaped = simulate(use_shaping=True)

# ==========================================
# 7. 結果のプロット
# ==========================================
plt.figure(figsize=(10, 8))

# 1. テーブルの絶対位置
plt.subplot(3, 1, 1)
plt.plot(t, r, 'k--', label='Reference')
plt.plot(t, xt_normal, 'r-', alpha=0.7, label='Table Pos (w/o Shaping)')
plt.plot(t, xt_shaped, 'b-', label='Table Pos (with Multi-mode Shaping)')
plt.title('Table Absolute Position (m2)')
plt.ylabel('Position [m]')
plt.legend(loc='lower right')
plt.grid(True)

# 2. 架台（設備本体）の振動
plt.subplot(3, 1, 2)
plt.plot(t, xb_normal, 'r-', alpha=0.7, label='Base Vibration (w/o Shaping)')
plt.plot(t, xb_shaped, 'b-', label='Base Vibration (with Multi-mode Shaping)')
plt.title('Machine Base Vibration (m3)')
plt.ylabel('Position [m]')
plt.legend(loc='lower right')
plt.grid(True)

# 3. 追従誤差（目標軌道 - テーブル絶対位置）
plt.subplot(3, 1, 3)
plt.plot(t, r - xt_normal, 'r-', alpha=0.7, label='Tracking Error (w/o Shaping)')
plt.plot(t, r - xt_shaped, 'b-', label='Tracking Error (with Multi-mode Shaping)')
plt.title('Table Tracking Error')
plt.xlabel('Time [s]')
plt.ylabel('Error [m]')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()