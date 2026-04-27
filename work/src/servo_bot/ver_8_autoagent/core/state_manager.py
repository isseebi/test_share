import streamlit as st
from typing import List, Dict, Any, Optional
from core.config import MODE_1IN

class StateManager:
    """st.session_state の型安全なラップ"""
    
    @staticmethod
    def init_state():
        """必要なセッション変数を初期化する"""
        default_states = {
            "messages": [],
            "sim_data": None,
            "saved_params": {},
            "prev_mode": None,
            "param_history": [],
            "pending_images": [],
            "prev_sim_data": None,
            "editing_idx": -1
        }
        for key, val in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = val

    @staticmethod
    def handle_mode_change(new_mode: str):
        """モード変更時の処理"""
        if new_mode != st.session_state.prev_mode:
            st.session_state.sim_data = None
            st.session_state.messages = []
            st.session_state.param_history = []
            st.session_state.prev_mode = new_mode
            st.session_state.pending_images = []
            st.session_state.prev_sim_data = None
            st.session_state.saved_params = {}
            
            # 不要な入力キーのみ削除
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith("input_")]
            for k in keys_to_delete:
                del st.session_state[k]

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return st.session_state.messages

    @messages.setter
    def messages(self, val: List[Dict[str, Any]]):
        st.session_state.messages = val

    @property
    def sim_data(self) -> Optional[Dict[str, Any]]:
        return st.session_state.sim_data

    @sim_data.setter
    def sim_data(self, val: Optional[Dict[str, Any]]):
        st.session_state.sim_data = val

    @property
    def prev_sim_data(self) -> Optional[Dict[str, Any]]:
        return st.session_state.prev_sim_data

    @prev_sim_data.setter
    def prev_sim_data(self, val: Optional[Dict[str, Any]]):
        st.session_state.prev_sim_data = val

    @property
    def saved_params(self) -> Dict[str, Any]:
        return st.session_state.saved_params

    @property
    def param_history(self) -> List[Dict[str, Any]]:
        return st.session_state.param_history

    @property
    def pending_images(self) -> List[str]:
        return st.session_state.pending_images

    @pending_images.setter
    def pending_images(self, val: List[str]):
        st.session_state.pending_images = val

    @property
    def editing_idx(self) -> int:
        return st.session_state.editing_idx

    @editing_idx.setter
    def editing_idx(self, val: int):
        st.session_state.editing_idx = val

    def update_saved_params(self, config_dict: Dict[str, Any]):
        """パラメータを保存し、モードごとのプレフィックス付きでも保存する"""
        suffix = "1in" if config_dict["mode"] == MODE_1IN else "2in"
        for key, val in config_dict.items():
            st.session_state.saved_params[key] = val
            if key != "mode":
                st.session_state.saved_params[f"{key}_{suffix}"] = val

    def add_param_history(self, user_request: str, parameters: Dict[str, Any]):
        st.session_state.param_history.append({
            "user_request": user_request,
            "parameters": parameters.copy()
        })
