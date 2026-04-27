import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from servo_sim import run_simulation, get_default_config
from analysis_drawing import calculate_stats, create_servo_figure
from LLM_call import ServoAssistantService, ConfigManager

# 日本語文字化け対策
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# ==========================================
# 0. 外部サービスの初期化 (キャッシュ)
# ==========================================
@st.cache_resource
def get_assistant_service():
    """LLMサービスを初期化し、セッション間で再利用する"""
    ConfigManager.load_env_from_file("api_key.txt")
    return ServoAssistantService(model_name="gemini")

# ==========================================
# 1. セッション状態の管理
# ==========================================
def init_session_state():
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

def handle_mode_change(new_mode):
    """モード変更時の処理。シミュレーション結果などをリセットするが、パラメータ(saved_params)は保持する"""
    if new_mode != st.session_state.prev_mode:
        st.session_state.sim_data = None
        # st.session_state.saved_params = {} # パラメータは保持したいので削除
        st.session_state.messages = []
        st.session_state.param_history = []
        st.session_state.prev_mode = new_mode
        st.session_state.pending_images = []
        st.session_state.prev_sim_data = None
        
        # 不要な入力キーのみ削除
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith("input_")]
        for k in keys_to_delete:
            del st.session_state[k]

# ==========================================
# 2. サイドバー・UI入力コンポーネント
# ==========================================
def param_input(label, key, default_val, step, format_str="%.2f"):
    """汎用的な数値入力UIコンポーネント"""
    current_val = float(st.session_state.saved_params.get(key, default_val))
    return st.sidebar.number_input(label, value=current_val, step=float(step), format=format_str, key=f"input_{key}")

def render_sidebar():
    """サイドバーを描画し、設定情報(config)と実行フラグを返す"""
    st.sidebar.header("1. シミュレーションモード選択")
    mode = st.sidebar.selectbox(
        "物理モデルを選択",
        ["1慣性系 (Single Inertia)", "2慣性系 (Two Inertia)"]
    )
    
    handle_mode_change(mode)
    defaults = get_default_config(mode)
    suffix = "1in" if mode == "1慣性系 (Single Inertia)" else "2in"

    st.sidebar.markdown("---")
    submit_top = st.sidebar.button("シミュレーション実行 / 更新", key="btn_submit_top", type="primary", use_container_width=True)
    
    config = {"mode": mode}
    
    # --- パラメータ設定の全項目をモードごとに分岐管理 ---
    if mode == "1慣性系 (Single Inertia)":
        st.sidebar.header("2. 物理パラメータ設定")
        st.sidebar.subheader("物理モデル")
        config["inertia"] = param_input("慣性 (Inertia)", f"inertia_{suffix}", defaults["inertia"], 0.1)
        config["friction"] = param_input("摩擦係数 (Friction)", f"friction_{suffix}", defaults["friction"], 0.1)

        st.sidebar.markdown("---")
        st.sidebar.subheader("制御パラメータ")
        st.sidebar.write("**PID 制御器**")
        config["kp"] = param_input("比例ゲイン (Kp)", f"kp_{suffix}", defaults["kp"], 1.0)
        config["ki"] = param_input("積分項ゲイン (Ki)", f"ki_{suffix}", defaults["ki"], 0.1)
        config["kd"] = param_input("微分項ゲイン (Kd)", f"kd_{suffix}", defaults["kd"], 0.1)
        config["limit"] = param_input("出力制限 (Limit)", f"limit_{suffix}", defaults["limit"], 10.0, format_str="%.1f")

        st.sidebar.write("**フィルタ**")
        config["use_cmd_lpf"] = st.sidebar.checkbox("指令 LPF", value=st.session_state.saved_params.get(f"use_cmd_lpf_{suffix}", defaults["use_cmd_lpf"]), key=f"input_use_cmd_lpf_{suffix}")
        config["cmd_cutoff"] = param_input("指令 LPF [Hz]", f"cmd_cutoff_{suffix}", defaults["cmd_cutoff"], 1.0)
        config["use_out_lpf"] = st.sidebar.checkbox("出力 LPF", value=st.session_state.saved_params.get(f"use_out_lpf_{suffix}", defaults["use_out_lpf"]), key=f"input_use_out_lpf_{suffix}")
        config["out_cutoff"] = param_input("出力 LPF [Hz]", f"out_cutoff_{suffix}", defaults["out_cutoff"], 1.0)

        st.sidebar.markdown("---")
        st.sidebar.subheader("3. シミュレーション環境")
        config["target_val"] = param_input("目標位置", f"target_val_{suffix}", defaults["target_val"], 10.0)
        config["dt"] = param_input("dt [s]", f"dt_{suffix}", defaults["dt"], 0.0001, format_str="%.4f")
        config["duration"] = param_input("時間 [s]", f"duration_{suffix}", defaults["duration"], 0.5)

    elif mode == "2慣性系 (Two Inertia)":
        st.sidebar.header("2. 物理パラメータ設定")
        st.sidebar.subheader("物理モデル")
        config["j_motor"] = param_input("モータ慣性 (Jm)", f"j_motor_{suffix}", defaults["j_motor"], 0.1)
        config["f_motor"] = param_input("モータ粘性摩擦 (Fm)", f"f_motor_{suffix}", defaults["f_motor"], 0.1)
        config["j_load"] = param_input("負荷慣性 (Jl)", f"j_load_{suffix}", defaults["j_load"], 0.1)
        config["f_load"] = param_input("負荷粘性摩擦 (Fl)", f"f_load_{suffix}", defaults["f_load"], 0.1)
        config["k_shaft"] = param_input("軸剛性 (Ks)", f"k_shaft_{suffix}", defaults["k_shaft"], 10.0, format_str="%.1f")
        config["c_shaft"] = param_input("軸減衰 (Cs)", f"c_shaft_{suffix}", defaults["c_shaft"], 0.5)

        st.sidebar.markdown("---")
        st.sidebar.subheader("制御パラメータ")
        st.sidebar.write("**PID 制御器**")
        config["kp"] = param_input("比例ゲイン (Kp)", f"kp_{suffix}", defaults["kp"], 1.0)
        config["ki"] = param_input("積分項ゲイン (Ki)", f"ki_{suffix}", defaults["ki"], 0.1)
        config["kd"] = param_input("微分項ゲイン (Kd)", f"kd_{suffix}", defaults["kd"], 0.1)
        config["limit"] = param_input("出力制限 (Limit)", f"limit_{suffix}", defaults["limit"], 10.0, format_str="%.1f")

        st.sidebar.write("**フィルタ**")
        config["use_cmd_lpf"] = st.sidebar.checkbox("指令 LPF", value=st.session_state.saved_params.get(f"use_cmd_lpf_{suffix}", defaults["use_cmd_lpf"]), key=f"input_use_cmd_lpf_{suffix}")
        config["cmd_cutoff"] = param_input("指令 LPF [Hz]", f"cmd_cutoff_{suffix}", defaults["cmd_cutoff"], 1.0)
        config["use_out_lpf"] = st.sidebar.checkbox("出力 LPF", value=st.session_state.saved_params.get(f"use_out_lpf_{suffix}", defaults["use_out_lpf"]), key=f"input_use_out_lpf_{suffix}")
        config["out_cutoff"] = param_input("出力 LPF [Hz]", f"out_cutoff_{suffix}", defaults["out_cutoff"], 1.0)

        st.sidebar.markdown("---")
        st.sidebar.subheader("3. シミュレーション環境")
        config["target_val"] = param_input("目標位置", f"target_val_{suffix}", defaults["target_val"], 10.0)
        config["dt"] = param_input("dt [s]", f"dt_{suffix}", defaults["dt"], 0.0001, format_str="%.4f")
        config["duration"] = param_input("時間 [s]", f"duration_{suffix}", defaults["duration"], 0.5)

    submit_bottom = st.sidebar.button("シミュレーション実行 / 更新", key="btn_submit_bottom", type="primary", use_container_width=True)
    is_submitted = submit_top or submit_bottom

    return config, is_submitted

# ==========================================
# 3. シミュレーション実行ロジック
# ==========================================
def execute_simulation(config):
    """シミュレーションを実行し、結果をセッションに保存する"""
    if st.session_state.sim_data is not None:
        st.session_state.prev_sim_data = st.session_state.sim_data

    with st.spinner("シミュレーション実行中..."):
        history = run_simulation(config)
    
    mode = config["mode"]
    inertia_info = (
        f"慣性: {config.get('inertia')}, 摩擦: {config.get('friction')}" 
        if mode == "1慣性系 (Single Inertia)" 
        else f"Jm: {config.get('j_motor')}, Jl: {config.get('j_load')}, Ks: {config.get('k_shaft')}"
    )

    st.session_state.sim_data = {
        "history": history,
        "mode": mode,
        "target_val": config["target_val"],
        "summary_data": {
            "kp": config["kp"], "ki": config["ki"], "kd": config["kd"], "limit": config["limit"],
            "inertia_info": inertia_info,
            "use_cmd_lpf": config["use_cmd_lpf"], "cmd_cutoff": config["cmd_cutoff"],
            "use_out_lpf": config["use_out_lpf"], "out_cutoff": config["out_cutoff"],
            "duration": config["duration"], "dt": config["dt"]
        }
    }


# ==========================================
# 5. 結果・グラフ描画
# ==========================================
def render_results():
    """シミュレーション結果（設定サマリー、グラフ、指標）を描画する"""
    data = st.session_state.sim_data
    history = data["history"]
    s_data = data["summary_data"]
    target_val = data["target_val"]
    mode_used = data["mode"]

    st.subheader("シミュレーション結果")
    st.info(f"**表示中の設定:** {mode_used}")
    
    col_cfg1, col_cfg2 = st.columns(2)
    # with col_cfg1:
    #     st.markdown(f"""
    #     **PID & 物理モデル:**
    #     - Kp: {s_data['kp']}, Ki: {s_data['ki']}, Kd: {s_data['kd']} (Limit: {s_data['limit']})
    #     - {s_data['inertia_info']}
    #     """)
    # with col_cfg2:
    #     st.markdown(f"""
    #     **フィルタ & 環境:**
    #     - 指令LPF: {'ON (' + str(s_data['cmd_cutoff']) + 'Hz)' if s_data['use_cmd_lpf'] else 'OFF'}
    #     - 出力LPF: {'ON (' + str(s_data['out_cutoff']) + 'Hz)' if s_data['use_out_lpf'] else 'OFF'}
    #     - 目標: {target_val}, 時間: {s_data['duration']}s
    #     """)

    stats = calculate_stats(history, target_val)
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
            prev_h = st.session_state.prev_sim_data["history"] if st.session_state.prev_sim_data else None
            # グラフ作成
            fig = create_servo_figure(
                history=history,
                target_val=target_val,
                mode=mode_used,
                use_cmd_lpf=s_data["use_cmd_lpf"],
                prev_history=prev_h,
                zoom=cfg["zoom"]
            )
            
            # 画像描画とチャット追加ボタン
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
                    if img_b64 not in st.session_state.pending_images:
                        st.session_state.pending_images.append(img_b64)
                        st.toast(f"{'拡大図' if cfg['zoom'] else '全体図'} を挿入しました")
                    else:
                        st.toast("この画像は既に挿入されています", icon="⚠️")

    prev_overshoot, prev_settling_time = None, None
    if st.session_state.prev_sim_data:
        ph = st.session_state.prev_sim_data["history"]
        pt = st.session_state.prev_sim_data["target_val"]
        p_stats = calculate_stats(ph, pt)
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


# ==========================================
# 6. チャットインターフェース
# ==========================================
def execute_llm_call(prompt_text, msg_content, config):
    """LLM呼び出しと履歴保存の共通処理"""
    st.session_state.messages.append({"role": "user", "content": msg_content})
    
    # パラメータを内部管理用に保存
    suffix = "1in" if config["mode"] == "1慣性系 (Single Inertia)" else "2in"
    for key, val in config.items():
        st.session_state.saved_params[key] = val
        if key != "mode":
            st.session_state.saved_params[f"{key}_{suffix}"] = val

    # LLMへの問い合わせ (サフィックスのないクリーンなパラメータを渡す)
    service = get_assistant_service()
    response = service.get_llm_response(
        st.session_state.messages, 
        config,
        st.session_state.param_history
    )
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.param_history.append({
        "user_request": prompt_text,
        "parameters": config.copy()
    })

def render_chat_interface(config):
    """チャット履歴の表示と入力処理を行う"""
    st.markdown("---")
    st.subheader("💬 AI アシスタント")
    
    chat_container = st.container(height=400)
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            col_msg, col_btns = st.columns([0.9, 0.1])
            with col_msg:
                with st.chat_message(message["role"]):
                    # 編集モード
                    if st.session_state.editing_idx == idx:
                        text_only = message["content"]
                        if isinstance(message["content"], list):
                            text_only = next((item["text"] for item in message["content"] if item["type"] == "text"), "")
                        
                        edit_text = st.text_area("メッセージを編集", value=text_only, key=f"area_{idx}")
                        c_e1, c_e2 = st.columns([0.2, 0.8])
                        
                        if c_e1.button("更新", key=f"save_{idx}"):
                            new_content = edit_text
                            if isinstance(message["content"], list):
                                new_content = [{"type": "text", "text": edit_text}] + [item for item in message["content"] if item["type"] != "text"]
                            
                            st.session_state.messages[idx]["content"] = new_content
                            st.session_state.messages = st.session_state.messages[:idx] # 更新直前までの履歴にクリップ
                            
                            execute_llm_call(edit_text, new_content, config)
                            
                            st.session_state.editing_idx = -1
                            st.rerun()
                            
                        if c_e2.button("キャンセル", key=f"cancel_{idx}"):
                            st.session_state.editing_idx = -1
                            st.rerun()
                    
                    # 通常表示モード
                    else:
                        if isinstance(message["content"], list):
                            for item in message["content"]:
                                if item["type"] == "text":
                                    st.markdown(item["text"])
                                elif item["type"] == "image_url":
                                    st.image(item["image_url"]["url"], width=300)
                        else:
                            st.markdown(message["content"])
            
            with col_btns:
                if message["role"] == "user" and st.session_state.editing_idx != idx:
                    st.write("")
                    b_col1, b_col2 = st.columns(2)
                    if b_col1.button("✏️", key=f"edit_btn_{idx}", help="編集"):
                        st.session_state.editing_idx = idx
                        st.rerun()
                    if b_col2.button("🗑️", key=f"del_btn_{idx}", help="削除"):
                        st.session_state.messages = st.session_state.messages[:idx]
                        st.rerun()

    # チャット入力に添付する画像の表示
    if st.session_state.pending_images:
        st.write("📎 挿入された画像:")
        cols = st.columns(len(st.session_state.pending_images) + 1)
        for i, img in enumerate(st.session_state.pending_images):
            with cols[i]:
                st.image(f"data:image/png;base64,{img}", width=100)
        if cols[-1].button("全てクリア"):
            st.session_state.pending_images = []
            st.rerun()

    # ユーザー入力
    if prompt := st.chat_input("パラメーターの調整方法について質問してください..."):
        msg_content = [{"type": "text", "text": prompt}]
        for img in st.session_state.pending_images:
            msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            
        execute_llm_call(prompt, msg_content, config)
        st.session_state.pending_images = []
        st.rerun()

# ==========================================
# 7. メイン処理
# ==========================================
def main():
    st.set_page_config(page_title="Servo Control Simulation", layout="wide")
    st.title("パラメータ調整アシストBot")

    # 1. 初期化
    init_session_state()

    # 2. UIレンダリング & パラメータ取得
    config, is_submitted = render_sidebar()

    # 3. シミュレーション実行
    if is_submitted:
        execute_simulation(config)

    # 4. 結果表示
    if st.session_state.sim_data:
        render_results()
    else:
        st.info("👈 左側のサイドバーでパラメータを設定し、「シミュレーション実行 / 更新」ボタンをクリックしてください。")
        st.write("\n" * 30)

    # 5. チャットインターフェース
    render_chat_interface(config)

if __name__ == "__main__":
    main()