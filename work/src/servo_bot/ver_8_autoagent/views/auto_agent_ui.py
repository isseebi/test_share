import streamlit as st
import io
import base64
from core.state_manager import StateManager
from core.simulator_bridge import SimulatorBridge
from services.llm_service import LLMService
from services.analysis_service import AnalysisService
from views.chat_ui import ChatUIView
from core.config import MODE_1IN

class AutoAgentUIView:
    """自動パラメータ調整エージェントの処理ループとUIコンポーネント"""
    
    def __init__(self, state_manager: StateManager, llm_service: LLMService):
        self.state = state_manager
        self.llm_service = llm_service
        self.chat_ui = ChatUIView(state_manager)

    def render_auto(self, current_config: dict):
        # 2回実行用のステップ管理
        if "auto_step" not in st.session_state:
            st.session_state.auto_step = 0
        if "auto_prompt" not in st.session_state:
            st.session_state.auto_prompt = ""

        # ステップ実行ロジック (UI更新の合間に実行される)
        if st.session_state.auto_step > 0:
            self._execute_step(current_config)

        # チャット送信ハンドラ
        def handle_chat_send(prompt_text, msg_content):
            # 最初のキックを行うだけ
            st.session_state.auto_prompt = prompt_text
            st.session_state.auto_step = 1
            st.rerun()

        self.chat_ui.render(on_send=handle_chat_send)

    def render_manual(self, current_config: dict):
        def handle_chat_send(prompt_text, msg_content):
            # 履歴追加
            self.state.messages.append({"role": "user", "content": msg_content})
            
            # パラメータ保存
            self.state.update_saved_params(current_config)
            
            # LLM呼び出し
            with st.spinner("AIが思考中..."):
                response = self.llm_service.get_response(
                    self.state.messages,
                    current_config,
                    self.state.param_history
                )
            
            # 回答追加
            self.state.messages.append({"role": "assistant", "content": response})
            
            # 履歴保存
            self.state.add_param_history(prompt_text, current_config)

        self.chat_ui.render(on_send=handle_chat_send)


    def _execute_step(self, current_config: dict):
        step = st.session_state.auto_step
        prompt_text = st.session_state.auto_prompt
        
        # --- 1. 画像生成 & ユーザーメッセージ構築 ---
        current_user_msg = [{"type": "text", "text": prompt_text}]

        # シミュレーション結果があれば拡大画像を自動で添付
        if self.state.sim_data:
            analysis = AnalysisService()
            data = self.state.sim_data
            fig = analysis.create_figure(
                history=data["history"],
                target_val=data["target_val"],
                mode=data["mode"],
                use_cmd_lpf=data["summary_data"]["use_cmd_lpf"],
                prev_history=self.state.prev_sim_data["history"] if self.state.prev_sim_data else None,
                zoom=True
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            current_user_msg.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        # チャット履歴に追加
        self.state.messages.append({"role": "user", "content": current_user_msg})
        self.state.update_saved_params(current_config)
        
        # --- 2. LLM呼び出し ---
        with st.spinner(f"AIが思考中... ({step}/2 回目)"):
            reason, p1_num, p1_val, p2_num, p2_val = self.llm_service.get_param_response(
                self.state.messages,
                current_config,
                self.state.param_history
            )
        
        # 回答追加
        self.state.messages.append({"role": "assistant", "content": reason})
        
        # --- 3. パラメータ反映 & 即時シミュレーション ---
        if p1_num > 0 or p2_num > 0:
            suffix = "1in" if current_config["mode"] == MODE_1IN else "2in"
            param_map = {
                1: f"kp_{suffix}", 2: f"ki_{suffix}", 3: f"kd_{suffix}",
                4: f"cmd_cutoff_{suffix}", 5: f"out_cutoff_{suffix}"
            }
            pending = st.session_state.get("pending_params", {})
            
            for p_num, p_val in [(p1_num, p1_val), (p2_num, p2_val)]:
                if p_num in param_map:
                    key = param_map[p_num]
                    val = float(p_val)
                    pending[key] = val
                    base_key = key.split("_")[0]
                    if base_key in current_config:
                        current_config[base_key] = val
                    elif "cutoff" in key:
                        current_config["cmd_cutoff" if "cmd" in key else "out_cutoff"] = val
                    
                    if p_num == 4:
                        pending[f"use_cmd_lpf_{suffix}"] = True
                        current_config["use_cmd_lpf"] = True
                    if p_num == 5:
                        pending[f"use_out_lpf_{suffix}"] = True
                        current_config["use_out_lpf"] = True
            
            st.session_state["pending_params"] = pending
            if self.state.sim_data is not None:
                self.state.prev_sim_data = self.state.sim_data
            self.state.sim_data = SimulatorBridge.execute(current_config)
        
        self.state.add_param_history(prompt_text, current_config)
        
        # 次のステップへ
        if step < 2:
            st.session_state.auto_step = step + 1
        else:
            st.session_state.auto_step = 0 # 終了
        
        st.rerun()
