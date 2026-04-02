import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 制御対象パラメータ (ボールねじ2慣性系モデル)
# ==========================================
# ※回転運動と並進運動を等価な直動系(質量・ばね)としてモデル化しています
m1 = 1.0     # モータ側の等価質量 [kg]
m2 = 1.0     # 負荷側(テーブル)の質量 [kg]
k = 2000.0   # ボールねじの等価剛性 [N/m]
c = 2.0      # ボールねじの粘性減衰係数 [N/(m/s)]

# システムの共振周波数と減衰比の計算
# 2慣性系の共振角周波数: omega_n = sqrt(k * (m1 + m2) / (m1 * m2))
# omega_n = np.sqrt(k * (m1 + m2) / (m1 * m2))
# zeta = c / (2 * np.sqrt(k * (m1 * m2) / (m1 + m2)))
omega_n = np.sqrt(k / m2)
zeta = c / (2 * np.sqrt(k * m2))
# ==========================================
# 2. パルス入力 (インプットシェーピング: ZV Shaper)
# ==========================================
# 振動を打ち消すための遅延時間と振幅を計算
omega_d = omega_n * np.sqrt(1 - zeta**2)
Td = np.pi / omega_d  # パルス間隔 (半周期)
K_shaper = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))

A1 = 1 / (1 + K_shaper)       # 最初のパルスの重み
A2 = K_shaper / (1 + K_shaper) # 遅延パルスの重み

# ==========================================
# 3. 制御器ゲイン (PI制御)
# ==========================================
Kp = 20000.0  
Kd = 400.0    
Ki = 0.0
# ==========================================
# 4. 目標軌道生成 (S字軌道: 加速度が台形)
# ==========================================
dt = 0.001
t = np.arange(0, 1.0, dt)
N = len(t)

r = np.zeros(N) # 目標位置
v = np.zeros(N) # 目標速度
a = np.zeros(N) # 目標加速度

# 0.1s ~ 0.5s でテーブルを移動させる
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
# 5. シェーピング軌道の作成 (パルス入力の畳み込み)
# ==========================================
shift = int(Td / dt)
r_shaped = np.zeros(N)
a_shaped = np.zeros(N)

for i in range(N):
    # 元の指令値 × A1
    r_shaped[i] = A1 * r[i]
    a_shaped[i] = A1 * a[i]
    
    # 遅延指令値 × A2 (ここで逆位相のパルスが入り振動を相殺)
    if i >= shift:
        r_shaped[i] += A2 * r[i - shift]
        a_shaped[i] += A2 * a[i - shift]

# ==========================================
# 6. シミュレーション実行関数
# ==========================================
def simulate(use_shaping=True):
    x1, v1 = 0.0, 0.0 # モータ位置, 速度
    x2, v2 = 0.0, 0.0 # 負荷(テーブル)位置, 速度
    err_sum = 0.0
    
    x2_log, u_log = [], []
    
    # シェーピングの有無で参照する指令値を変更
    ref_r = r_shaped if use_shaping else r
    ref_a = a_shaped if use_shaping else a
    
    for i in range(N):
        # --- (A) フィードフォアード制御 ---
        # 剛体としての理想的な必要推力 (F = m * a)
        u_ff = (m1 + m2) * ref_a[i]
        
        # --- (B) PI制御 (フィードバック) ---
        # モータの位置情報をフィードバックして誤差を計算
        err = ref_r[i] - x1
        err_sum += err * dt
        u_fb = Kp * err + Ki * err_sum

        err = ref_r[i] - x1
        # モーター速度 v1 を使ってダンピングを効かせる
        u_fb = Kp * err - Kd * v1 

        # 全制御入力
        u = u_ff + u_fb
        
        # --- (C) 制御対象のダイナミクス (2慣性系) ---
        # ボールねじの弾性による内部伝達力
        F_spring = k * (x1 - x2) + c * (v1 - v2)
        
        # 運動方程式
        a1 = (u - F_spring) / m1  # モータ側の加速度
        a2 = F_spring / m2        # 負荷(テーブル)側の加速度
        
        # オイラー法による積分
        v1 += a1 * dt
        x1 += v1 * dt
        v2 += a2 * dt
        x2 += v2 * dt
        
        x2_log.append(x2)
        u_log.append(u)
        
    return np.array(x2_log)

# 実行
x2_normal = simulate(use_shaping=False)
x2_shaped = simulate(use_shaping=True)

# ==========================================
# 7. 結果のプロット
# ==========================================
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, r, 'k--', label='Reference (Target)')
plt.plot(t, x2_normal, 'r-', alpha=0.7, label='Table Pos (w/o Pulse Shaping)')
plt.plot(t, x2_shaped, 'b-', label='Table Pos (with Pulse Shaping)')
plt.title('Ball Screw Table Position (Load Side)')
plt.ylabel('Position [m]')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
# 誤差を拡大して表示
plt.plot(t, r - x2_normal, 'r-', alpha=0.7, label='Tracking Error (w/o Shaping)')
plt.plot(t, r - x2_shaped, 'b-', label='Tracking Error (with Shaping)')
plt.title('Tracking Error')
plt.xlabel('Time [s]')
plt.ylabel('Error [m]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()