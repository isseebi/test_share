import streamlit as st
from core.state_manager import StateManager
from core.simulator_bridge import SimulatorBridge
from services.llm_service import LLMService
from views.sidebar import SidebarView
from views.dashboard import DashboardView
from views.chat_ui import ChatUIView
from views.optimization_ui import OptimizationView


def simulator_page():
    """シミュレータメインページ"""
    st.title("パラメータ調整アシストBot")

    # 1. 初期化
    state_manager = StateManager()
    state_manager.init_state()
    
    llm_service = LLMService()
    
    sidebar = SidebarView(state_manager)
    dashboard = DashboardView(state_manager)
    chat_ui = ChatUIView(state_manager)
    opt_ui = OptimizationView()

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


    # 5. チャットインターフェース
    def handle_chat_send(prompt_text, msg_content):
        # 履歴追加
        state_manager.messages.append({"role": "user", "content": msg_content})
        
        # パラメータ保存
        state_manager.update_saved_params(config)
        
        # LLM呼び出し
        with st.spinner("AIが思考中..."):
            response = llm_service.get_response(
                state_manager.messages,
                config,
                state_manager.param_history
            )
        
        # 回答追加
        state_manager.messages.append({"role": "assistant", "content": response})
        
        # 履歴保存
        state_manager.add_param_history(prompt_text, config)


    chat_ui.render(on_send=handle_chat_send)
    
    # 6. ベイズ最適化
    opt_ui.render(state_manager, config["mode"], config, SimulatorBridge.execute)

def dummy_page():
    """ダミーの2ページ目"""
    st.title("ドキュメント / 設定")
    st.write("ここにドキュメントや追加の設定項目を表示する予定です。")
    st.info("これは st.navigation を使用したマルチページ構成のテストです。")

def main():
    st.set_page_config(page_title="Servo Control Simulation", layout="wide")

    # ページ定義
    pages = [
        st.Page(simulator_page, title="対話型パラメータ調整Bot", icon="🤖"),
        # st.Page(dummy_page, title="ドキュメント", icon="📄"),
    ]

    # ナビゲーションの実行
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
