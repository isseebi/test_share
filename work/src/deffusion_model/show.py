import numpy as np

# データの読み込みテスト
data = np.load("/home/isseebi/Desktop/user/Reinforcement_learning/study/OpenManipulation/src/diffusion_dataset.npy")
print(data.shape) # 例: (300, 2)
print(data[0])    # 最初のxy座標