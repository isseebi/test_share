import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class PIDController:
    """PID制御器クラス（出力制限・アンチワインドアップ付き）"""
    def __init__(self, kp: float, ki: float, kd: float, dt: float, output_limit: float = np.inf):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limit = output_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        
        # 積分項の更新
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # 出力飽和とアンチワインドアップ処理（積分項のクランプ）
        if output > self.output_limit:
            output = self.output_limit
            self.integral -= error * self.dt # 飽和時は積分を進めない
        elif output < -self.output_limit:
            output = -self.output_limit
            self.integral -= error * self.dt

        self.prev_error = error
        return output

class LowPassFilter:
    """1次遅れローパスフィルタ"""
    def __init__(self, cutoff_freq: float, dt: float):
        self.dt = dt
        self.prev_output = 0.0
        self.alpha = 0.0
        self.set_cutoff(cutoff_freq)

    def set_cutoff(self, cutoff_freq: float) -> None:
        if cutoff_freq <= 0:
            self.alpha = 1.0
        else:
            # 時定数 tau = 1 / (2 * pi * fc)
            tau = 1.0 / (2.0 * np.pi * cutoff_freq)
            self.alpha = self.dt / (tau + self.dt)

    def filter(self, u: float) -> float:
        output = self.alpha * u + (1.0 - self.alpha) * self.prev_output
        self.prev_output = output
        return output

class SingleInertiaModel:
    """1慣性系の物理モデル（剛体モデル）"""
    def __init__(self, inertia: float, friction: float, dt: float):
        self.inertia = inertia
        self.friction = friction
        self.dt = dt
        self.velocity = 0.0
        self.position = 0.0

    @property
    def motor_pos(self) -> float:
        return self.position

    @property
    def load_pos(self) -> float:
        return self.position

    def update(self, torque: float) -> Tuple[float, float]:
        """状態を1ステップ更新し、(モータ位置, 負荷位置)を返す"""
        acceleration = (torque - self.friction * self.velocity) / self.inertia
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        return self.motor_pos, self.load_pos

class TwoInertiaModel:
    """2慣性系の物理モデル（モータと負荷がバネ要素で結合されたモデル）"""
    def __init__(self, j_motor: float, j_load: float, k_shaft: float, c_shaft: float, 
                 f_motor: float, f_load: float, dt: float):
        self.j_motor = j_motor   # モータ側慣性
        self.j_load = j_load     # 負荷側慣性
        self.k_shaft = k_shaft   # 軸剛性
        self.c_shaft = c_shaft   # 軸の内部減衰（粘性）
        self.f_motor = f_motor   # モータ側粘性摩擦
        self.f_load = f_load     # 負荷側粘性摩擦
        self.dt = dt
        
        self._pos_m = 0.0
        self._vel_m = 0.0
        self._pos_l = 0.0
        self._vel_l = 0.0

    @property
    def motor_pos(self) -> float:
        return self._pos_m

    @property
    def load_pos(self) -> float:
        return self._pos_l

    def update(self, torque: float) -> Tuple[float, float]:
        """状態を1ステップ更新し、(モータ位置, 負荷位置)を返す"""
        # モータと負荷の間のねじれトルク（結合力）
        coupling_torque = (self.k_shaft * (self._pos_m - self._pos_l) + 
                           self.c_shaft * (self._vel_m - self._vel_l))
        
        # 各慣性の加速度計算
        acc_m = (torque - coupling_torque - self.f_motor * self._vel_m) / self.j_motor
        acc_l = (coupling_torque - self.f_load * self._vel_l) / self.j_load
        
        # 状態更新（オイラー法）
        self._vel_m += acc_m * self.dt
        self._pos_m += self._vel_m * self.dt
        self._vel_l += acc_l * self.dt
        self._pos_l += self._vel_l * self.dt
        
        return self.motor_pos, self.load_pos

class ServoSimulator:
    """サーボ制御シミュレータークラス"""
    def __init__(self, model, controller: PIDController, 
                 cmd_filter: Optional[LowPassFilter] = None, 
                 out_filter: Optional[LowPassFilter] = None):
        self.model = model
        self.controller = controller
        self.cmd_filter = cmd_filter
        self.out_filter = out_filter

    def run(self, duration: float, target_val: float) -> dict:
        dt = self.controller.dt
        steps = int(duration / dt)
        
        # 事前に配列を確保（appendよりも高速・明示的）
        history = {
            "time": np.linspace(0, duration, steps),
            "target_raw": np.full(steps, target_val),
            "target_filtered": np.zeros(steps),
            "motor_pos": np.zeros(steps),
            "load_pos": np.zeros(steps)
        }

        for i in range(steps):
            # A. 指令フィルタ処理
            smoothed_setpoint = self.cmd_filter.filter(target_val) if self.cmd_filter else target_val
            
            # B. フィードバック取得とPID計算
            # 共通インターフェース化したため、属性の有無を気にする必要がなくなりました
            current_fdbk = self.model.motor_pos
            control_signal = self.controller.compute(smoothed_setpoint, current_fdbk)
            
            # C. 出力フィルタ処理
            if self.out_filter:
                control_signal = self.out_filter.filter(control_signal)
            
            # D. プラント（物理モデル）の更新
            pos_m, pos_l = self.model.update(control_signal)
            
            # 記録
            history["target_filtered"][i] = smoothed_setpoint
            history["motor_pos"][i] = pos_m
            history["load_pos"][i] = pos_l

        return history