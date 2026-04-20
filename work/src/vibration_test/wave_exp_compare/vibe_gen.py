import numpy as np
import matplotlib.pyplot as plt
import os  # フォルダ作成やパス操作のために追加

class VibrationSimulator:
    """
    スライダとノズルの2自由度系振動シミュレータ
    物理演算、外乱生成、振動補償（インプットシェーピング）、グラフ描画を管理します。
    """
    def __init__(self, mode="low_freq_vibration", with_compensation=False, shaper_wn=None, shaper_zeta=None):
        """
        コンストラクタ：モード指定による物理パラメータと制御設定の初期化
        """
        self.mode = mode
        self.with_compensation = with_compensation
        
        # --- デフォルト値の初期化（各モードで上書きされる） ---
        self.M = self.m = self.dt = self.time_max = 0.0
        self.wn = self.zeta = self.Kp = self.Kd = 0.0
        self.name = "Simulation"
        
        # 振動補償（インプットシェーピング）の設定
        self.shaper_wn = None
        self.shaper_zeta = None

        # 外乱の設定
        self.dist_type = "none" 
        self.dist_amp = 0.0   

        # --- モードごとのパラメータ適用 ---
        self._apply_mode(mode)

        # --- 手動指定がある場合は上書き (Trueのときのみ有効) ---
        if self.with_compensation:
            if shaper_wn is not None:
                self.shaper_wn = shaper_wn
            if shaper_zeta is not None:
                self.shaper_zeta = shaper_zeta

        # シェーピングを使用するかどうかの最終判定
        self.use_shaping = (self.with_compensation and self.shaper_wn is not None)
        


        # --- 指定されたモードの分類関数を適用 ---
        # self._apply_mode(mode)

        self.use_shaping = (self.shaper_wn is not None)
        
        # 物理定数の計算（バネ定数 k, 減衰係数 c）
        self.k = self.m * (self.wn**2)
        self.c = 2 * self.m * self.zeta * self.wn
        
        # 結果格納用
        self.results_base = None    
        self.results_shaped = None  
    
    def get_parameters(self):
        return {
            "mode": self.mode,
            "name": self.name,
            "M": self.M,
            "m": self.m,
            "k": self.k,
            "c": self.c,
            "dt": self.dt,
            "time_max": self.time_max,
            "wn": self.wn,
            "zeta": self.zeta,
            "Kp": self.Kp,
            "Kd": self.Kd,
            "dist_type": self.dist_type,
            "dist_amp": self.dist_amp,
            "use_shaping": self.use_shaping,
            "shaper_wn": self.shaper_wn,
            "shaper_zeta": self.shaper_zeta
        }

    # ====================================================
    # 分類関数群（パラメータ設定メソッド）
    # ====================================================
    def _apply_mode(self, mode):
        """モード名に応じて各設定メソッドを呼び出す"""
        if mode == "low_freq_vibration":
            self._low_freq_vibration()
        elif mode == "high_freq_vibration":
            self._high_freq_vibration()
        # elif mode == "low_freq_overshoot":
        #     self._low_freq_overshoot()
        # elif mode == "high_freq_overshoot":
        #     self._high_freq_overshoot()
        # elif mode == "low_amplitude_overshoot":
        #     self._low_amplitude_overshoot()
        # elif mode == "high_amplitude_overshoot":
        #     self._high_amplitude_overshoot()
        elif mode == "white_noise_model":
            self._white_noise_model()
        elif mode == "pulse_wave_model":
            self._pulse_wave_model()
        elif mode == "custom_equation_model":
            self._custom_equation_model()
        else:
            raise ValueError(f"Unknown simulation mode: {mode}")

    def _low_freq_vibration(self):
        """分類：低周波振動モデル（ゆったりと大きく揺れる）"""
        self.M, self.m, self.dt, self.time_max = 1.0, 0.1, 0.001, 1.0
        self.wn, self.zeta, self.Kp, self.Kd = 100.0, 0.03, 150.0, 10.0
        self.name = "Low-Frequency Vibration Mode"
        if self.with_compensation:
            self.shaper_wn, self.shaper_zeta = 30.0, 0.01

    def _high_freq_vibration(self):
        """分類：高周波振動モデル（細かく速く揺れる）"""
        self.M, self.m, self.dt, self.time_max = 1.0, 0.1, 0.001, 0.5
        self.wn, self.zeta, self.Kp, self.Kd = 300.0, 0.05, 150.0, 15.0
        self.name = "High-Frequency Vibration Mode"
        if self.with_compensation:
            self.shaper_wn, self.shaper_zeta = 250.0, 0.01

    # def _low_freq_overshoot(self):
    #     """分類：低周波オーバーシュートモデル（低剛性で目標値を大きく越える）"""
    #     self.M, self.m, self.dt, self.time_max = 1.0, 0.4, 0.001, 1.0
    #     self.wn, self.zeta, self.Kp, self.Kd = 25.0, 0.02, 120.0, 5.0 # Kp高くKd低い
    #     self.name = "Low-Freq High-Overshoot Mode"
    #     if self.with_compensation:
    #         self.shaper_wn, self.shaper_zeta = 25.0, 0.02

    # def _high_freq_overshoot(self):
    #     """分類：高周波オーバーシュートモデル（鋭い立ち上がりと速い振動）"""
    #     self.M, self.m, self.dt, self.time_max = 1.0, 0.05, 0.0001, 1.0
    #     self.wn, self.zeta, self.Kp, self.Kd = 150.0, 0.02, 400.0, 10.0
    #     self.name = "High-Freq Sharp-Overshoot Mode"
    #     if self.with_compensation:
    #         self.shaper_wn, self.shaper_zeta = 150.0, 0.02

    # def _low_amplitude_overshoot(self):
    #     """分類：低振幅オーバーシュートモデル（微小な行き過ぎ）"""
    #     self.M, self.m, self.dt, self.time_max = 1.0, 0.2, 0.001, 1.0
    #     self.wn, self.zeta, self.Kp, self.Kd = 60.0, 0.1, 80.0, 15.0 # 減衰を高めに設定
    #     self.name = "Small-Amplitude Overshoot Mode"
    #     if self.with_compensation:
    #         self.shaper_wn, self.shaper_zeta = 60.0, 0.1

    # def _high_amplitude_overshoot(self):
    #     """分類：高振幅オーバーシュートモデル（制御が不安定に近い状態）"""
    #     self.M, self.m, self.dt, self.time_max = 1.0, 0.1, 0.001, 1.0
    #     self.wn, self.zeta, self.Kp, self.Kd = 20.0, 0.01, 150.0, 2.0 # 極端に低いKd
    #     self.name = "Wild-Amplitude Overshoot Mode"
    #     if self.with_compensation:
    #         self.shaper_wn, self.shaper_zeta = 20.0, 0.01

    def _white_noise_model(self):
        """分類：ホワイトノイズモデル（定常的なランダム外乱の影響）"""
        self.M, self.m, self.dt, self.time_max = 1.0, 0.1, 0.001, 1.0
        self.wn, self.zeta, self.Kp, self.Kd = 100.0, 0.05, 150.0, 10.0
        self.dist_type, self.dist_amp = "noise", 3.0
        self.name = "Steady-State Noise Mode"
        if self.with_compensation:
            self.shaper_wn, self.shaper_zeta = 50.0, 0.01

    def _pulse_wave_model(self):
        """分類：パルス波モデル（突発的な衝撃外乱）"""
        self.M, self.m, self.dt, self.time_max = 1.0, 0.1, 0.001, 1.0
        self.wn, self.zeta, self.Kp, self.Kd = 100.0, 0.05, 150.0, 10.0
        self.dist_type, self.dist_amp = "pulse", 20.0
        self.name = "Sudden Impact Pulse Mode"
        if self.with_compensation:
            self.shaper_wn, self.shaper_zeta = 50.0, 0.01

    def _custom_equation_model(self):
        """分類：物理方程式モデル（任意の減衰と固有振動数を持つ波を付加）"""
        self.M, self.m, self.dt, self.time_max = 1.0, 0.1, 0.001, 1.0
        self.wn, self.zeta, self.Kp, self.Kd = 100.0, 0.05, 150.0, 10.0
        self.dist_type, self.dist_amp = "equation", 10.0
        self.name = "Custom Equation Wave Mode"
        if self.with_compensation:
            self.shaper_wn, self.shaper_zeta = 50.0, 0.01

    # ====================================================
    # 演算・描画処理群
    # ====================================================
    def _generate_disturbance(self, n_steps):
        """ノズル（手先）に加わる外乱信号を生成"""
        dist = np.zeros(n_steps)
        t = np.arange(0, self.time_max, self.dt)
        
        if self.dist_type == "noise":
            dist = np.random.normal(0, self.dist_amp, n_steps)
        elif self.dist_type == "pulse":
            pulse_start, pulse_end = 0.2, 0.3
            dist[(t >= pulse_start) & (t <= pulse_end)] = self.dist_amp
        elif self.dist_type == "equation":
            # 任意の物理方程式に従う波形を追加
            eq_wn = 40.0    # 独自の固有振動数
            eq_zeta = 0.1  # 独自の減衰率
            eq_wd = eq_wn * np.sqrt(1 - eq_zeta**2)
            
            # 物理方程式：A * e^(-zeta * wn * t) * sin(wd * t)
            dist = self.dist_amp * np.exp(-eq_zeta * eq_wn * t) * np.sin(eq_wd * t)
            
        return dist

    def _calculate(self, use_shaper=False):
        """物理演算コア：運動方程式を解く"""
        t = np.arange(0, self.time_max, self.dt)
        n_steps = len(t)
        dist_signal = self._generate_disturbance(n_steps)
        
        # --- インプットシェーピングのパラメータ計算 ---
        if use_shaper:
            wd = self.shaper_wn * np.sqrt(1 - self.shaper_zeta**2)
            K_val = np.exp(-self.shaper_zeta * np.pi / np.sqrt(1 - self.shaper_zeta**2))
            A1, A2 = 1 / (1 + K_val), K_val / (1 + K_val)
            t2 = np.pi / wd
        else:
            A1, A2, t2 = 1.0, 0.0, 0.0

        # 変数初期化
        xs, vs = np.zeros(n_steps), np.zeros(n_steps) # スライダ
        xn, vn = np.zeros(n_steps), np.zeros(n_steps) # ノズル
        u = np.zeros(n_steps)                         # 制御力
        target_pos = 1.0                              # 目標位置

        for i in range(1, n_steps):
            curr_t = i * self.dt
            
            # 1. 目標値の生成（シェーピング適用時は入力を分割）
            curr_target = target_pos * A1 if (use_shaper and curr_t < t2) else target_pos
            if use_shaper and curr_t >= t2: 
                curr_target = target_pos * (A1 + A2)

            # 2. 制御力の計算（PD制御）
            F_ctrl = self.Kp * (curr_target - xs[i-1]) - self.Kd * vs[i-1]
            
            # 3. 相互作用力の計算
            F_int = self.k * (xs[i-1] - xn[i-1]) + self.c * (vs[i-1] - vn[i-1])
            
            # 4. 加速度の計算
            a_s = (F_ctrl - F_int) / self.M
            a_n = (F_int + dist_signal[i]) / self.m
            
            # 5. 積分
            vs[i] = vs[i-1] + a_s * self.dt
            xs[i] = xs[i-1] + vs[i] * self.dt
            vn[i] = vn[i-1] + a_n * self.dt
            xn[i] = xn[i-1] + vn[i] * self.dt
            u[i] = F_ctrl # 制御入力を記録

        return {'t': t, 'xn': xn, 'xs': xs, 'u': u, 'deflection': xn - xs}

    def run(self):
        """シミュレーション実行"""
        self.results_base = self._calculate(use_shaper=False)
        if self.use_shaping:
            self.results_shaped = self._calculate(use_shaper=True)
        return self.results_base

    def _compute_fft(self, data):
        """FFT解析：揺れの周波数成分を抽出"""
        sig = data['deflection']
        n = len(sig)
        sig_detrended = sig - np.mean(sig)
        f = np.fft.fftfreq(n, self.dt)
        m = np.abs(np.fft.fft(sig_detrended))[f > 0] * (2.0 / n)
        return f[f > 0], m

    def plot(self):
        """結果の可視化（4段構成）"""
        fig, axs = plt.subplots(4, 1, figsize=(10, 13))
        fig.suptitle(f"Vibration Analysis: {self.name}", fontsize=15)

        # 描画対象の設定
        plots = [(self.results_base, 'gray', 0.5, 'Base (No Shaper)')]
        if self.use_shaping:
            plots.append((self.results_shaped, 'red', 1.0, 'Suppressed (Shaped)'))
        else:
            plots = [(self.results_base, 'blue', 1.0, 'Simulation Result')]

        for res, color, alpha, label in plots:
            # 1段目: ノズル絶対位置
            axs[0].plot(res['t'], res['xn'], color=color, alpha=alpha, label=label)
            # 2段目: 相対変位（揺れ）
            axs[1].plot(res['t'], res['deflection'], color=color, alpha=alpha, label=label)
            # 3段目: FFTスペクトル
            freqs, mags = self._compute_fft(res)
            axs[2].plot(freqs, mags, color=color, alpha=alpha, label=label)
            # 4段目: 制御入力
            axs[3].plot(res['t'], res['u'], color=color, alpha=alpha, label=f"Control Force ({label})")

        # グラフ装飾
        axs[0].set_title("Nozzle Position (Absolute)"); axs[0].set_ylabel("Position [m]")
        axs[1].set_title("Nozzle Deflection (xn - xs)"); axs[1].set_ylabel("Deflection [m]")
        axs[2].set_title("Frequency Spectrum (FFT)"); axs[2].set_xlim(0, 100); axs[2].set_ylabel("Amplitude")
        
        # 4段目の設定
        axs[3].set_title("Control Input (Slider Force)"); axs[3].set_ylabel("Force [N]"); axs[3].set_xlabel("Time [s]")
        
        for ax in axs: 
            ax.legend(loc='upper right')
            ax.grid(True, linestyle=':')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # --- 画像の保存処理 ---
        # ファイル名から不適切な文字を除去
        clean_name = self.name.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(f"{clean_name}.png")
        plt.savefig(save_path)

        plt.show() # 画面表示
        plt.close(fig) # メモリ解放


# --- 実行 ---
if __name__ == "__main__":
    # --- 実行するシミュレーションモードのリスト ---
    sim_cases = [
        "low_freq_vibration",
        "high_freq_vibration",
        "white_noise_model",
        "pulse_wave_model",
        "custom_equation_model"
    ]
    
    print("====================================================")
    print("   全振動分類シミュレーション シーケンス開始")
    print("   (グラフウィンドウを閉じると次のケースに進みます)")
    print("====================================================")

    # # 各ケースをループで順番に実行
    # for i, mode_name in enumerate(sim_cases, 1):
    #     # 1. 指定したモード（補償あり）でインスタンスを生成
    #     sim_instance = VibrationSimulator(mode=mode_name, with_compensation=False)
        
    #     print(f"\n[{i}/{len(sim_cases)}] 実行中: {sim_instance.name}")
        
    #     # 2. 計算の実行
    #     sim_instance.run()
        
    #     # 3. グラフの表示
    #     sim_instance.plot()

    # print("\n====================================================")
    # print("   すべてのシミュレーションが正常に終了しました。")
    # print("====================================================")

    sim2 = VibrationSimulator(
        mode="custom_equation_model", 
        with_compensation=False, 
        shaper_wn=100.0,     # デフォルト(30.0)を50.0に上書き
        shaper_zeta=0.05     # デフォルト(0.03)を0.1に上書き
    )
    sim2.run()
    params = sim2.get_parameters()
    print(f"Shaper Params: wn={params['shaper_wn']}, zeta={params['shaper_zeta']}")
    sim2.plot()