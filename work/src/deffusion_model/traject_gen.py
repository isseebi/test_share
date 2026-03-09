import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d

class RandomPathGenerator:
    """
    ランダムな通過点を通る滑らかな曲線を生成し、
    一定間隔（step_size）でサンプリングするクラス。
    """
    def __init__(self, step_size=0.1, num_points_range=(3, 4)):
        self.step_size = step_size
        self.num_points = np.random.randint(num_points_range[0], num_points_range[1])
        
        # 生成されたデータの保持用
        self.raw_points = None
        self.path_fine = {}    # 高密度な曲線データ
        self.path_approx = {}  # step_sizeでリサンプリングしたデータ
        self.total_length = 0

    def generate(self):
        """経路の生成からサンプリングまでを一括で実行する"""
        self._generate_random_points()
        self._create_smooth_base()
        self._resample_by_step_size()
        return self.path_approx['x'], self.path_approx['y']

    def _generate_random_points(self):
        """1. 通過点の生成（原点を含む）"""
        random_coords = np.random.uniform(-1, 1, size=(self.num_points, 2))
        self.raw_points = {
            'x': np.insert(random_coords[:, 0], 0, 0),
            'y': np.insert(random_coords[:, 1], 0, 0)
        }

    def _create_smooth_base(self, resolution=1000):
        """2. スプライン補間による滑らかなベース曲線の作成"""
        x, y = self.raw_points['x'], self.raw_points['y']
        t = np.linspace(0, 1, len(x))
        t_fine = np.linspace(0, 1, resolution)
        k = min(3, len(x) - 1)

        spl_x = make_interp_spline(t, x, k=k)
        spl_y = make_interp_spline(t, y, k=k)

        self.path_fine['x'] = spl_x(t_fine)
        self.path_fine['y'] = spl_y(t_fine)

    def _resample_by_step_size(self):
        """3. 累積道のりに基づく一定間隔サンプリング"""
        x_f, y_f = self.path_fine['x'], self.path_fine['y']
        
        # 累積距離（弧長）の計算
        dx = np.diff(x_f)
        dy = np.diff(y_f)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        self.total_length = cumulative_dist[-1]

        # サンプリング位置の決定
        sampling_distances = np.arange(0, self.total_length, self.step_size)
        if sampling_distances[-1] < self.total_length - (self.step_size / 2):
            sampling_distances = np.append(sampling_distances, self.total_length)

        # 座標の逆引き補間
        interp_x = interp1d(cumulative_dist, x_f, kind='linear')
        interp_y = interp1d(cumulative_dist, y_f, kind='linear')

        self.path_approx['x'] = interp_x(sampling_distances)
        self.path_approx['y'] = interp_y(sampling_distances)

    def plot(self):
        """結果の可視化"""
        plt.figure(figsize=(8, 6))
        plt.plot(self.raw_points['x'], self.raw_points['y'], 'ro', label='Control Points')
        plt.plot(self.path_fine['x'], self.path_fine['y'], 'g--', alpha=0.5, label='Original Spline')
        plt.plot(self.path_approx['x'], self.path_approx['y'], 'b-o', markersize=4, label='Resampled Path')
        plt.title(f"Path Generation (Step Size: {self.step_size})")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


# --- 使用例 ---
if __name__ == "__main__":
    generator = RandomPathGenerator(step_size=0.15)
    x_approx, y_approx = generator.generate()
    
    print(f"生成されたポイント数: {len(x_approx)}")
    print(f"経路の全長: {generator.total_length:.2f}")
    
    generator.plot()