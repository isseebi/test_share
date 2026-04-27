import numpy as np
import time
import optuna
from typing import Dict, Any, Callable, List, Tuple
# from core.config import MODE_1IN # 必要に応じてインポートを有効化してください

class OptimizationService:
    """ベイズ最適化（Optuna使用）のロジックを管理"""
    
    @staticmethod
    def run_optimization(
        mode: str,
        base_config: Dict[str, Any],
        simulator_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        on_progress: Callable[[int, int, Dict[str, Any]], None],
        n_trials: int = 10
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optunaを用いたベイズ最適化を実行する。
        """
        # Optunaのデフォルトの標準出力ログが進行状況表示の邪魔にならないよう抑制
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        history = []

        # 探索範囲の設定
        search_space = {
            "kp": (base_config.get("kp", 1.0) * 0.5, base_config.get("kp", 1.0) * 2.0),
            "ki": (base_config.get("ki", 0.0) * 0.5, base_config.get("ki", 20.0) * 2.0), # Ki=0の場合に備えて少し幅を持たせる
            "kd": (base_config.get("kd", 0.0) * 0.5, base_config.get("kd", 20.0) * 2.0),
        }

        # フィルタが有効な場合は、遮断周波数も探索対象に含める
        if base_config.get("use_cmd_lpf"):
            current_val = base_config.get("cmd_cutoff", 100.0)
            search_space["cmd_cutoff"] = (max(0.1, current_val * 0.1), min(500.0, current_val * 5.0))
        
        if base_config.get("use_out_lpf"):
            current_val = base_config.get("out_cutoff", 100.0)
            search_space["out_cutoff"] = (max(0.1, current_val * 0.1), min(500.0, current_val * 5.0))

        # 目的関数（Optunaが各トライアルで呼び出す関数）
        def objective(trial: optuna.Trial) -> float:
            trial_params = base_config.copy()
            current_trial_data = {}
            
            # ベイズ最適化アルゴリズムに基づき、次に試すパラメータを提案
            for param, (low, high) in search_space.items():
                val = trial.suggest_float(param, low, high)
                trial_params[param] = val
                current_trial_data[param] = val

            # シミュレーション実行
            result = simulator_fn(trial_params)
            
            # スコア計算（目標値との偏差の二乗和など）
            m_pos = np.array(result["history"]["motor_pos"])
            target = trial_params["target_val"]
            
            # 安定性や追従性を評価するためのスコア（単純なMSEの他、発散防止のため大きな値を返す）
            if np.any(np.isnan(m_pos)) or np.any(np.isinf(m_pos)) or np.max(np.abs(m_pos)) > target * 10:
                return 1e10
                
            score = float(np.mean((m_pos - target)**2))
            
            current_trial_data["score"] = score
            history.append(current_trial_data)

            # 進捗報告
            on_progress(trial.number + 1, n_trials, current_trial_data)
            
            # 演出用のスリープ（実際のシミュレーションが重い場合は不要になります）
            time.sleep(0.05) 

            return score

        # OptunaのStudyを作成
        # TPE（Tree-structured Parzen Estimator）というベイズ最適化アルゴリズムがデフォルトで利用されます
        # 偏差を「最小化」したいので direction="minimize" を指定
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(), 
            direction="minimize"
        )
        
        # 最適化の実行
        study.optimize(objective, n_trials=n_trials)

        # 最良のパラメータを取得して返す
        best_params = base_config.copy()
        best_params.update(study.best_params)

        return best_params, history