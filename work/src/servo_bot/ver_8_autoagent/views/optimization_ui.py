import streamlit as st
import pandas as pd
from typing import Dict, Any, Callable
from core.config import MODE_1IN, MODE_2IN
from services.optimization_service import OptimizationService

class OptimizationView:
    """ベイズ最適化UIコンポーネント"""
    
    def __init__(self):
        # サービスを初期化
        self.service = OptimizationService()

    def render(self, state_manager: Any, mode: str, current_config: Dict[str, Any], simulator_bridge_fn: Callable):
        st.markdown("---")
        
        with st.expander("🎯 ベイズ最適化 (パラメータ自動調整)", expanded=False):
            st.write("PIDゲインを自動的に最適化します。")
            
            # 試行回数の設定
            n_trials = st.number_input("試行回数", min_value=5, max_value=50, value=10, key="opt_n_trials")
            
            # --- 1. 最適化の実行 ---
            if st.button("最適化を開始", key="run_opt_btn", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, data):
                    progress_bar.progress(current / total)
                    status_text.text(f"試行 {current}/{total}: Kp={data['kp']:.2f}, Score={data['score']:.4f}")

                # 最適化サービスを実行
                # ここでエラーが出る場合は、引数の型や simulator_bridge_fn の実装を確認
                best_params, history = self.service.run_optimization(
                    mode=mode,
                    base_config=current_config,
                    simulator_fn=simulator_bridge_fn,
                    on_progress=update_progress,
                    n_trials=n_trials
                )
                
                # 結果を一時保存（リラン後も保持するため）
                st.session_state["last_opt_result"] = {
                    "best_params": best_params,
                    "history": history,
                    "mode": mode
                }
                st.success("最適化が完了しました！下を確認してください。")

            # --- 2. 結果表示と反映処理 ---
            if "last_opt_result" in st.session_state:
                res = st.session_state["last_opt_result"]
                # モードが切り替わっていたらクリア
                if res["mode"] != mode:
                    del st.session_state["last_opt_result"]
                    st.rerun()
                    return

                best_params = res["best_params"]
                suffix = "1in" if mode == MODE_1IN else "2in"

                st.markdown("### 🏆 最適化結果")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**推奨パラメータ:**")
                    params_text = f"Kp: {best_params['kp']:.4f}\nKi: {best_params['ki']:.4f}\nKd: {best_params['kd']:.4f}"
                    if "cmd_cutoff" in best_params:
                        params_text += f"\n指令 LPF: {best_params['cmd_cutoff']:.2f} Hz"
                    if "out_cutoff" in best_params:
                        params_text += f"\n出力 LPF: {best_params['out_cutoff']:.2f} Hz"
                    st.code(params_text)
                    
                    def apply_callback(params, sfx, mgr):
                        # パラメータリスト
                        p_list = ["kp", "ki", "kd", "cmd_cutoff", "out_cutoff"]
                        update_dict = {"mode": mode}
                        
                        for p in p_list:
                            if p in params:
                                val = float(params[p])
                                st.session_state[f"input_{p}_{sfx}"] = val
                                update_dict[p] = val
                        
                        # StateManager の内部データも同期
                        mgr.update_saved_params(update_dict)
                        
                        # 反映後は一時結果を削除
                        if "last_opt_result" in st.session_state:
                            del st.session_state["last_opt_result"]

                    st.button(
                        "このパラメータをサイドバーに反映", 
                        type="primary", 
                        key="apply_opt_btn",
                        on_click=apply_callback,
                        args=(best_params, suffix, state_manager),
                        use_container_width=True
                    )

                with col2:
                    st.write("**試行履歴:**")
                    st.dataframe(pd.DataFrame(res["history"]), height=180, use_container_width=True)
                
                if st.button("結果をクリア", key="clear_opt_res"):
                    del st.session_state["last_opt_result"]
                    st.rerun()