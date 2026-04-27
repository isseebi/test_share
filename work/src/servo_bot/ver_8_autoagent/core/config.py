from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

# シミュレーションモードの定数
MODE_1IN = "1慣性系 (Single Inertia)"
MODE_2IN = "2慣性系 (Two Inertia)"

@dataclass
class SimulationConfig:
    mode: str
    kp: float
    ki: float
    kd: float
    limit: float
    dt: float
    duration: float
    target_val: float
    use_cmd_lpf: bool = False
    cmd_cutoff: float = 100.0
    use_out_lpf: bool = False
    out_cutoff: float = 100.0
    
    # 1慣性系用
    inertia: Optional[float] = None
    friction: Optional[float] = None
    
    # 2慣性系用
    j_motor: Optional[float] = None
    f_motor: Optional[float] = None
    j_load: Optional[float] = None
    f_load: Optional[float] = None
    k_shaft: Optional[float] = None
    c_shaft: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

DEFAULT_CONFIGS = {
    MODE_1IN: {
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
    MODE_2IN: {
        "j_motor": 1.0,
        "f_motor": 0.5,
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
