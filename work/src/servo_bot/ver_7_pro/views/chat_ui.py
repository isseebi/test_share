import streamlit as st
from typing import Dict, Any, List, Callable
from core.state_manager import StateManager

class ChatUIView:
    """チャットインターフェースコンポーネント"""
    
    def __init__(self, state_manager: StateManager):
        self.state = state_manager

    def render(self, on_send: Callable[[str, List[Dict[str, Any]]], None]):
        st.markdown("---")
        
        with st.expander("💬 AI アシスタント", expanded=True):
            chat_container = st.container(height=400)
            with chat_container:
                for idx, message in enumerate(self.state.messages):
                    col_msg, col_btns = st.columns([0.9, 0.1])
                    with col_msg:
                        with st.chat_message(message["role"]):
                            if self.state.editing_idx == idx:
                                self._render_edit_mode(idx, message, on_send)
                            else:
                                self._render_message_content(message)
                    
                    with col_btns:
                        self._render_action_buttons(idx, message)

            self._render_pending_images()
            self._render_chat_input(on_send)

    def _render_message_content(self, message: Dict[str, Any]):
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"], width=300)
        else:
            st.markdown(message["content"])

    def _render_edit_mode(self, idx: int, message: Dict[str, Any], on_send: Callable):
        text_only = message["content"]
        if isinstance(message["content"], list):
            text_only = next((item["text"] for item in message["content"] if item["type"] == "text"), "")
        
        edit_text = st.text_area("メッセージを編集", value=text_only, key=f"area_{idx}")
        c_e1, c_e2 = st.columns([0.2, 0.8])
        
        if c_e1.button("更新", key=f"save_{idx}"):
            new_content = edit_text
            if isinstance(message["content"], list):
                new_content = [{"type": "text", "text": edit_text}] + [item for item in message["content"] if item["type"] != "text"]
            
            # 更新処理
            self.state.messages = self.state.messages[:idx]
            on_send(edit_text, new_content)
            self.state.editing_idx = -1
            st.rerun()
            
        if c_e2.button("キャンセル", key=f"cancel_{idx}"):
            self.state.editing_idx = -1
            st.rerun()

    def _render_action_buttons(self, idx: int, message: Dict[str, Any]):
        if message["role"] == "user" and self.state.editing_idx != idx:
            st.write("")
            b_col1, b_col2 = st.columns(2)
            if b_col1.button("✏️", key=f"edit_btn_{idx}", help="編集"):
                self.state.editing_idx = idx
                st.rerun()
            if b_col2.button("🗑️", key=f"del_btn_{idx}", help="削除"):
                self.state.messages = self.state.messages[:idx]
                st.rerun()

    def _render_pending_images(self):
        if self.state.pending_images:
            st.write("📎 挿入された画像:")
            cols = st.columns(len(self.state.pending_images) + 1)
            for i, img in enumerate(self.state.pending_images):
                with cols[i]:
                    st.image(f"data:image/png;base64,{img}", width=100)
            if cols[-1].button("全てクリア"):
                self.state.pending_images = []
                st.rerun()

    def _render_chat_input(self, on_send: Callable):
        if prompt := st.chat_input("パラメーターの調整方法について質問してください..."):
            msg_content = [{"type": "text", "text": prompt}]
            for img in self.state.pending_images:
                msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            
            on_send(prompt, msg_content)
            self.state.pending_images = []
            st.rerun()
