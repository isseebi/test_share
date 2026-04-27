import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sample_sim_core import PIDController, LowPassFilter, SingleInertiaModel ,TwoInertiaModel ,ServoSimulator



def plot_results(history: dict, title: str = "Servo Simulation Results") -> None:
    plt.figure(figsize=(10, 6))
    time = history["time"]
    plt.plot(time, history["target_raw"], 'k:', label='Target (Raw)', alpha=0.5)
    plt.plot(time, history["target_filtered"], 'g--', label='Target (Filtered)')
    plt.plot(time, history["motor_pos"], 'b-', label='Motor Position', alpha=0.7)
    plt.plot(time, history["load_pos"], 'r-', label='Load Position', alpha=0.9)
    
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Position / Angle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------
# デフォルトパラメータ設定
# -----------------------------------------------------
DEFAULT_CONFIGS = {
    "1慣性系 (Single Inertia)": {
        "inertia": 1.5,
        "friction": 0.5,
        "kp": 15.0,
        "ki": 5.0,
        "kd": 4.0,
        "limit": 500.0,
        "cmd_cutoff": 100.0,
        "out_cutoff": 100.0,
        "target_val": 100.0,
        "dt": 0.001,
        "duration": 5.0,
        "use_cmd_lpf": False,
        "use_out_lpf": False
    },
    "2慣性系 (Two Inertia)": {
        "j_motor": 1.0,
        "f_motor": 0.5,
        "j_m": 1.0, # 互換性のため
        "f_m": 0.5, 
        "j_load": 4.0,
        "f_load": 0.5,
        "k_shaft": 200.0,
        "c_shaft": 5.0,
        "kp": 500.0,
        "ki": 10.0,
        "kd": 50.0,
        "limit": 5000.0,
        "cmd_cutoff": 100.0,
        "out_cutoff": 100.0,
        "target_val": 100.0,
        "dt": 0.001,
        "duration": 5.0,
        "use_cmd_lpf": True,
        "use_out_lpf": True
    }
}

def get_default_config(mode: str) -> dict:
    """指定されたモードのデフォルト設定を取得する"""
    return DEFAULT_CONFIGS.get(mode, DEFAULT_CONFIGS["1慣性系 (Single Inertia)"]).copy()

def run_simulation(config: dict) -> dict:
    """
    シミュレーションを実行するためのエントリポイント。
    config辞書からパラメータを受け取り、sample_sim_coreを使用して計算を行う。
    """
    mode = config.get("mode")
    dt = config.get("dt", 0.001)
    target_val = config.get("target_val", 100.0)
    duration = config.get("duration", 5.0)

    # PID制御器の初期化
    controller = PIDController(
        kp=config.get("kp", 15.0),
        ki=config.get("ki", 5.0),
        kd=config.get("kd", 4.0),
        dt=dt,
        output_limit=config.get("limit", 500.0)
    )

    # 物理モデルの初期化
    if mode == "1慣性系 (Single Inertia)":
        model = SingleInertiaModel(
            inertia=config.get("inertia", 1.5),
            friction=config.get("friction", 0.5),
            dt=dt
        )
    else:
        model = TwoInertiaModel(
            j_motor=config.get("j_motor", 1.0),
            j_load=config.get("j_load", 4.0),
            k_shaft=config.get("k_shaft", 200.0),
            c_shaft=config.get("c_shaft", 5.0),
            f_motor=config.get("f_motor", 0.5),
            f_load=config.get("f_load", 0.5),
            dt=dt
        )

    # フィルタの初期化
    cmd_filter = None
    if config.get("use_cmd_lpf"):
        cmd_filter = LowPassFilter(cutoff_freq=config.get("cmd_cutoff", 100.0), dt=dt)

    out_filter = None
    if config.get("use_out_lpf"):
        out_filter = LowPassFilter(cutoff_freq=config.get("out_cutoff", 100.0), dt=dt)

    # シミュレータの構築と実行
    simulator = ServoSimulator(
        model=model,
        controller=controller,
        cmd_filter=cmd_filter,
        out_filter=out_filter
    )

    history = simulator.run(duration=duration, target_val=target_val)
    return history

if __name__ == "__main__":
    # テスト用実行例
    test_config = {
        "mode": "1慣性系 (Single Inertia)",
        "dt": 0.001,
        "target_val": 100.0,
        "duration": 5.0,
        "kp": 15.0, "ki": 5.0, "kd": 4.0, "limit": 500.0,
        "inertia": 1.5, "friction": 0.5
    }
    
    print("Running test simulation...")
    history = run_simulation(test_config)
    print(f"Simulation completed. Steps: {len(history['time'])}")
    
    # 2慣性系のテスト
    test_config_2 = {
        "mode": "2慣性系 (Two Inertia)",
        "dt": 0.001,
        "target_val": 100.0,
        "duration": 2.0,
        "kp": 500.0, "ki": 10.0, "kd": 50.0, "limit": 5000.0,
        "j_motor": 1.0, "j_load": 4.0, "k_shaft": 2000.0, "c_shaft": 5.0,
        "f_motor": 0.5, "f_load": 0.5,
        "use_cmd_lpf": True, "cmd_cutoff": 1.0,
        "use_out_lpf": True, "out_cutoff": 15.0
    }
    print("Running test simulation (2-inertia)...")
    history2 = run_simulation(test_config_2)
    print(f"Simulation completed. Steps: {len(history2['time'])}")
    
    # 結果のプロット (以前の関数を使用)
    plot_results(history2, title="Two Inertia Model with LPF (Test)")