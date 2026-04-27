import streamlit as st
from typing import Dict, Any, Tuple
from core.config import MODE_1IN, MODE_2IN, DEFAULT_CONFIGS
from core.state_manager import StateManager

class SidebarView:
    """サイドバーUIコンポーネント"""
    
    def __init__(self, state_manager: StateManager):
        self.state = state_manager

    def render(self) -> Tuple[Dict[str, Any], bool]:
        st.sidebar.header("1. シミュレーションモード選択")
        mode = st.sidebar.selectbox(
            "物理モデルを選択",
            [MODE_1IN, MODE_2IN]
        )
        
        self.state.handle_mode_change(mode)
        defaults = DEFAULT_CONFIGS[mode]
        suffix = "1in" if mode == MODE_1IN else "2in"

        st.sidebar.markdown("---")
        submit_top = st.sidebar.button("シミュレーション実行 / 更新", key="btn_submit_top", type="primary", use_container_width=True)
        
        config = {"mode": mode}
        
        # --- 2. 物理パラメータ設定 ---
        st.sidebar.markdown("---")
        st.sidebar.header("2. 物理モデル設定")
        if mode == MODE_1IN:
            config["inertia"] = self._param_input("慣性 (Inertia)", f"inertia_{suffix}", defaults["inertia"], 0.1)
            config["friction"] = self._param_input("摩擦係数 (Friction)", f"friction_{suffix}", defaults["friction"], 0.1)
        elif mode == MODE_2IN:
            config["j_motor"] = self._param_input("モータ慣性 (Jm)", f"j_motor_{suffix}", defaults["j_motor"], 0.1)
            config["f_motor"] = self._param_input("モータ粘性摩擦 (Fm)", f"f_motor_{suffix}", defaults["f_motor"], 0.1)
            config["j_load"] = self._param_input("負荷慣性 (Jl)", f"j_load_{suffix}", defaults["j_load"], 0.1)
            config["f_load"] = self._param_input("負荷粘性摩擦 (Fl)", f"f_load_{suffix}", defaults["f_load"], 0.1)
            config["k_shaft"] = self._param_input("軸剛性 (Ks)", f"k_shaft_{suffix}", defaults["k_shaft"], 10.0, format_str="%.1f")
            config["c_shaft"] = self._param_input("軸減衰 (Cs)", f"c_shaft_{suffix}", defaults["c_shaft"], 0.5)

        # --- 3. 制御パラメータ設定 (PID + Filter) ---
        st.sidebar.markdown("---")
        st.sidebar.header("3. 制御器・フィルタ設定")
        st.sidebar.subheader("PID 制御器")
        config["kp"] = self._param_input("比例ゲイン (Kp)", f"kp_{suffix}", defaults["kp"], 1.0)
        config["ki"] = self._param_input("積分項ゲイン (Ki)", f"ki_{suffix}", defaults["ki"], 0.1)
        config["kd"] = self._param_input("微分項ゲイン (Kd)", f"kd_{suffix}", defaults["kd"], 0.1)
        config["limit"] = self._param_input("出力制限 (Limit)", f"limit_{suffix}", defaults["limit"], 10.0, format_str="%.1f")

        st.sidebar.subheader("フィルタ (LPF)")
        config["use_cmd_lpf"] = st.sidebar.checkbox("指令 LPF 有効", value=self.state.saved_params.get(f"use_cmd_lpf_{suffix}", defaults["use_cmd_lpf"]), key=f"input_use_cmd_lpf_{suffix}")
        config["cmd_cutoff"] = self._param_input("指令 LPF [Hz]", f"cmd_cutoff_{suffix}", defaults["cmd_cutoff"], 1.0)
        
        config["use_out_lpf"] = st.sidebar.checkbox("出力 LPF 有効", value=self.state.saved_params.get(f"use_out_lpf_{suffix}", defaults["use_out_lpf"]), key=f"input_use_out_lpf_{suffix}")
        config["out_cutoff"] = self._param_input("出力 LPF [Hz]", f"out_cutoff_{suffix}", defaults["out_cutoff"], 1.0)

        # --- 4. シミュレーション環境設定 ---
        st.sidebar.markdown("---")
        st.sidebar.header("4. シミュレーション設定")
        config["target_val"] = self._param_input("目標位置", f"target_val_{suffix}", defaults["target_val"], 10.0)
        config["dt"] = self._param_input("dt [s]", f"dt_{suffix}", defaults["dt"], 0.0001, format_str="%.4f")
        config["duration"] = self._param_input("時間 [s]", f"duration_{suffix}", defaults["duration"], 0.5)

        submit_bottom = st.sidebar.button("シミュレーション実行 / 更新", key="btn_submit_bottom", type="primary", use_container_width=True)
        is_submitted = submit_top or submit_bottom

        return config, is_submitted

    def _param_input(self, label, key, default_val, step, format_str="%.2f"):
        st_key = f"input_{key}"
        # セッション状態に未登録なら、保存済みパラメータまたはデフォルト値で初期化
        if st_key not in st.session_state:
            st.session_state[st_key] = float(self.state.saved_params.get(key, default_val))
        
        # value引数は指定せず、keyのみを指定することでセッション状態を唯一のソースにする
        return st.sidebar.number_input(label, step=float(step), format=format_str, key=st_key)
