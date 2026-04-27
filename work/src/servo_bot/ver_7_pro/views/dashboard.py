import streamlit as st
import io
import base64
from typing import Dict, Any
from core.state_manager import StateManager
from services.analysis_service import AnalysisService

class DashboardView:
    """結果表示（グラフ、メトリクス）コンポーネント"""
    
    def __init__(self, state_manager: StateManager):
        self.state = state_manager
        self.analysis = AnalysisService()

    def render(self):
        data = self.state.sim_data
        if not data:
            st.info("👈 左側のサイドバーでパラメータを設定し、「シミュレーション実行 / 更新」ボタンをクリックしてください。")
            st.write("\n" * 30)
            return

        history = data["history"]
        s_data = data["summary_data"]
        target_val = data["target_val"]
        mode_used = data["mode"]

        st.subheader("シミュレーション結果")
        st.info(f"**表示中の設定:** {mode_used}")
        
        stats = self.analysis.get_stats(history, target_val)
        overshoot_val = stats["overshoot"]
        settling_time = stats["settling_time"]

        # グラフ描画
        col_plot1, col_plot2 = st.columns(2)
        plot_configs = [
            {"col": col_plot1, "zoom": False},
            {"col": col_plot2, "zoom": True}
        ]
        
        for cfg in plot_configs:
            with cfg["col"]:
                prev_h = self.state.prev_sim_data["history"] if self.state.prev_sim_data else None
                fig = self.analysis.create_figure(
                    history=history,
                    target_val=target_val,
                    mode=mode_used,
                    use_cmd_lpf=s_data["use_cmd_lpf"],
                    prev_history=prev_h,
                    zoom=cfg["zoom"]
                )
                
                col_plot, col_ins = st.columns([0.92, 0.1])
                with col_plot:
                    st.pyplot(fig, use_container_width=True)
                with col_ins:
                    st.write("")
                    btn_key = "zoom" if cfg["zoom"] else "overview"
                    if st.button("➕", key=f"ins_btn_{btn_key}", help="チャットに挿入"):
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight')
                        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        if img_b64 not in self.state.pending_images:
                            self.state.pending_images.append(img_b64)
                            st.toast(f"{'拡大図' if cfg['zoom'] else '全体図'} を挿入しました")
                        else:
                            st.toast("この画像は既に挿入されています", icon="⚠️")

        # メトリクス表示
        self._render_metrics(stats)

    def _render_metrics(self, stats: Dict[str, Any]):
        overshoot_val = stats["overshoot"]
        settling_time = stats["settling_time"]
        
        prev_overshoot, prev_settling_time = None, None
        if self.state.prev_sim_data:
            ph = self.state.prev_sim_data["history"]
            pt = self.state.prev_sim_data["target_val"]
            p_stats = self.analysis.get_stats(ph, pt)
            prev_overshoot, prev_settling_time = p_stats["overshoot"], p_stats["settling_time"]

        col_res2, col_res3, _ = st.columns(3)
        with col_res2:
            delta_over = overshoot_val - prev_overshoot if prev_overshoot is not None else None
            st.metric("オーバーシュート", f"{overshoot_val:.2f}", delta=f"{delta_over:.2f}" if delta_over is not None else None, delta_color="inverse")
            if prev_overshoot is not None:
                st.caption(f"前回: {prev_overshoot:.2f}")

        with col_res3:
            delta_st = settling_time - prev_settling_time if (settling_time is not None and prev_settling_time is not None) else None
            val_text = f"{(settling_time if settling_time is not None else 0):.3f} s" if settling_time is not None else "未収束"
            st.metric("制定時間 (±2%)", val_text, delta=f"{delta_st:.3f} s" if delta_st is not None else None, delta_color="inverse")
            if prev_settling_time is not None:
                p_st_val = prev_settling_time if prev_settling_time is not None else 0
                st.caption(f"前回: {p_st_val:.3f} s" if prev_settling_time is not None else "前回: 未収束")
