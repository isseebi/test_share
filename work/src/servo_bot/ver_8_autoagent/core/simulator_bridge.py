from typing import Dict, Any
from servo_sim import run_simulation
from core.config import MODE_1IN, MODE_2IN

class SimulatorBridge:
    """外部シミュレータ(servo_sim)との仲介"""
    
    @staticmethod
    def execute(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """シミュレーションを実行し、UI向けに整形されたデータを返す"""
        history = run_simulation(config_dict)
        
        mode = config_dict["mode"]
        if mode == MODE_1IN:
            inertia_info = f"慣性: {config_dict.get('inertia')}, 摩擦: {config_dict.get('friction')}"
        elif mode == MODE_2IN:
            inertia_info = f"Jm: {config_dict.get('j_motor')}, Jl: {config_dict.get('j_load')}, Ks: {config_dict.get('k_shaft')}"
        else:
            inertia_info = "不明"

        return {
            "history": history,
            "mode": mode,
            "target_val": config_dict["target_val"],
            "summary_data": {
                "kp": config_dict["kp"], 
                "ki": config_dict["ki"], 
                "kd": config_dict["kd"], 
                "limit": config_dict["limit"],
                "inertia_info": inertia_info,
                "use_cmd_lpf": config_dict["use_cmd_lpf"], 
                "cmd_cutoff": config_dict["cmd_cutoff"],
                "use_out_lpf": config_dict["use_out_lpf"], 
                "out_cutoff": config_dict["out_cutoff"],
                "duration": config_dict["duration"], 
                "dt": config_dict["dt"]
            }
        }
