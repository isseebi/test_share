import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d

class RandomPathGenerator:
    """
    ランダムな通過点付近を通る滑らかな曲線を生成し、
    一定間隔（step_size）でサンプリングするクラス。
    """
    # s: 平滑化係数。大きいほど滑らか（直線的）になり、0だと元の点すべてを通る。
    def __init__(self, step_size=0.1, num_points_range=(3, 4), smoothing_factor=0.5):
        self.step_size = step_size
        self.num_points = np.random.randint(num_points_range[0], num_points_range[1])
        self.s = smoothing_factor  # 尖りを抑えるためのパラメータ
        
        self.raw_points = None
        self.path_fine = {}
        self.path_approx = {}
        self.total_length = 0

    def generate(self):
        self._generate_random_points()
        self._create_smooth_base()
        self._resample_by_step_size()
        return self.path_approx['x'], self.path_approx['y']

    def _generate_random_points(self):
        # 点が少なすぎるとスプラインが引けないため最小3点程度を推奨
        random_coords = np.random.uniform(-1, 1, size=(self.num_points, 2))
        self.raw_points = {
            'x': np.insert(random_coords[:, 0], 0, 0),
            'y': np.insert(random_coords[:, 1], 0, 0)
        }

    def _create_smooth_base(self, resolution=1000):
        """3. splprepを使用して平滑化スプラインを作成"""
        x, y = self.raw_points['x'], self.raw_points['y']
        
        # splprepは2次元以上の点群を直接扱える
        # s: 平滑化の強さ。点数が多い場合は s=len(x) 程度が目安
        # k: スプラインの次数（3 = 3次スプライン）
        tck, u = splprep([x, y], s=self.s, k=3)
        
        # 0から1の間で細かくサンプリング
        u_fine = np.linspace(0, 1, resolution)
        new_points = splev(u_fine, tck)

        self.path_fine['x'] = new_points[0]
        self.path_fine['y'] = new_points[1]

    def _resample_by_step_size(self):
        x_f, y_f = self.path_fine['x'], self.path_fine['y']
        
        dx = np.diff(x_f)
        dy = np.diff(y_f)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        self.total_length = cumulative_dist[-1]

        sampling_distances = np.arange(0, self.total_length, self.step_size)
        if len(sampling_distances) > 0 and sampling_distances[-1] < self.total_length - (self.step_size / 2):
            sampling_distances = np.append(sampling_distances, self.total_length)

        interp_x = interp1d(cumulative_dist, x_f, kind='linear')
        interp_y = interp1d(cumulative_dist, y_f, kind='linear')

        self.path_approx['x'] = interp_x(sampling_distances)
        self.path_approx['y'] = interp_y(sampling_distances)

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.raw_points['x'], self.raw_points['y'], 'ro', label='Control Points (Target)')
        plt.plot(self.path_fine['x'], self.path_fine['y'], 'g--', alpha=0.5, label='Smoothed Spline')
        plt.plot(self.path_approx['x'], self.path_approx['y'], 'b-o', markersize=4, label='Resampled Path')
        plt.title(f"Smoothed Path (s={self.s}, step={self.step_size})")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

# --- 実行 ---
if __name__ == "__main__":
    # smoothing_factor (s) を大きくすると、より「丸い」経路になります
    generator = RandomPathGenerator(step_size=0.1, smoothing_factor=1.5)
    x_approx, y_approx = generator.generate()
    generator.plot()