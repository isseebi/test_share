import streamlit as st
import io
import base64
from core.state_manager import StateManager
from core.simulator_bridge import SimulatorBridge
from services.llm_service import LLMService
from services.analysis_service import AnalysisService
from views.sidebar import SidebarView
from views.dashboard import DashboardView
from views.chat_ui import ChatUIView
from views.optimization_ui import OptimizationView
from views.auto_agent_ui import AutoAgentUIView
from core.config import MODE_1IN, MODE_2IN


def apply_pending_updates():
    """保留中のパラメータ更新を反映する"""
    if "pending_params" in st.session_state:
        for key, val in st.session_state["pending_params"].items():
            st.session_state[f"input_{key}"] = val
        del st.session_state["pending_params"]

def param_adj_demo_page():
    """パラメータ調整デモページ"""
    apply_pending_updates()
    st.title("パラメータ調整アシストBot")

    # 1. 初期化
    state_manager = StateManager()
    state_manager.init_state()
    
    llm_service = LLMService()
    
    sidebar = SidebarView(state_manager)
    dashboard = DashboardView(state_manager)
    chat_ui = ChatUIView(state_manager)
    opt_ui = OptimizationView()

    auto_agent_ui = AutoAgentUIView(state_manager, llm_service)

    # 2. UIレンダリング & パラメータ取得
    config, is_submitted = sidebar.render()

    # 3. シミュレーション実行
    if is_submitted:
        if state_manager.sim_data is not None:
            state_manager.prev_sim_data = state_manager.sim_data

        with st.spinner("シミュレーション実行中..."):
            sim_result = SimulatorBridge.execute(config)
            state_manager.sim_data = sim_result

    # 4. 結果表示
    dashboard.render()

    # 5. エージェント機能（手動調整チャット）のレンダリング
    auto_agent_ui.render_manual(config)
    
    # 6. ベイズ最適化
    opt_ui.render(state_manager, config["mode"], config, SimulatorBridge.execute)

def param_adj_auto_page():
    """自動パラメータ調整エージェントページ"""
    apply_pending_updates()
    st.title("パラメータ調整アシストBot")

    # 1. 初期化
    state_manager = StateManager()
    state_manager.init_state()
    
    llm_service = LLMService()
    
    sidebar = SidebarView(state_manager)
    dashboard = DashboardView(state_manager)
    
    auto_agent_ui = AutoAgentUIView(state_manager, llm_service)

    # 2. UIレンダリング & パラメータ取得
    current_config, is_submitted = sidebar.render()

    # 3. シミュレーション実行
    if is_submitted:
        if state_manager.sim_data is not None:
            state_manager.prev_sim_data = state_manager.sim_data

        with st.spinner("シミュレーション実行中..."):
            sim_result = SimulatorBridge.execute(current_config)
            state_manager.sim_data = sim_result

    # 4. 結果表示
    dashboard.render()

    # 5. エージェント機能（会話・ループ）のレンダリング
    auto_agent_ui.render_auto(current_config)

def main():
    st.set_page_config(page_title="Servo Control Simulation", layout="wide")

    # ページ定義
    pages = [
        st.Page(param_adj_demo_page, title="対話型パラメータ調整Bot"),
        st.Page(param_adj_auto_page, title="自動パラメータ調整エージェント"),
    ]

    # ナビゲーションの実行
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
