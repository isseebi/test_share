import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('vibration_data.csv')

# グラフ作成（2段構成）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 上段: 変位 (Displacement)
ax1.plot(df['time(s)'], df['displacement(m)'], color='tab:blue', marker='o', linestyle='-', label='displacement(m)')
ax1.set_ylabel('Displacement (m)')
ax1.set_title('Displacement over Time')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# 下段: カートと先端の位置 (Cart/Tip Position)
ax2.plot(df['time(s)'], df['cart_x(m)'], color='tab:red', marker='x', linestyle='--', label='cart_x(m)')
ax2.plot(df['time(s)'], df['tip_x(m)'], color='tab:green', marker='s', linestyle='-', label='tip_x(m)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('Cart and Tip Position over Time')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('vibration_analysis.png')
plt.show()