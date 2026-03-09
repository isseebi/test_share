import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock1D(nn.Module):
    """1D CNNの残差ブロック（時間と条件の埋め込みを受け取る）"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # 特徴量を抽出する畳み込み層
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.act = nn.GELU()
        # GroupNorm はバッチサイズに依存せず安定した学習が可能
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)

    def forward(self, x, emb):
        # x: (B, dim, L), emb: (B, dim, 1)
        # 埋め込みベクトルを足し合わせる（FiLMの簡易版）
        h = x + emb
        h = self.norm1(h)
        h = self.act(self.conv1(h))
        h = self.norm2(h)
        h = self.conv2(h)
        # 残差接続（Skip Connection）
        return x + h

class ConditionalTrajectoryDiffusion1DCNN(nn.Module):
    def __init__(self, seq_len=128, feature_dim=2, cond_dim=8, hidden_dim=128, num_blocks=4):
        """
        :param seq_len: 軌道のステップ数
        :param feature_dim: 座標の次元数 (x, y = 2)
        :param cond_dim: 条件の次元数 (始点+経由点1+経由点2+終点 = 8)
        :param hidden_dim: CNNの内部チャネル数
        :param num_blocks: ResNetブロックの数
        """
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # --- 1. 時間の埋め込み ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # --- 2. 条件(Cond)の埋め込み ---
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # --- 3. メインの1D CNN構造 ---
        # 入力を (B, feature_dim, L) から (B, hidden_dim, L) へ変換
        self.input_proj = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        
        # ResNetブロックを重ねる
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size=3) for _ in range(num_blocks)
        ])
        
        # 出力を (B, hidden_dim, L) から元の (B, feature_dim, L) へ戻す
        self.output_proj = nn.Conv1d(hidden_dim, feature_dim, kernel_size=1)

    def forward(self, x, time, cond):
        # PyTorchのConv1dは入力を (Batch, Channel, Length) の順で受け取るため転置する
        # x.shape: (B, 128, 2) -> (B, 2, 128)
        x = x.transpose(1, 2)
        
        # 埋め込みベクトルの計算
        t_emb = self.time_mlp(time)  # (B, hidden_dim)
        c_emb = self.cond_mlp(cond)  # (B, hidden_dim)
        
        # 時間と条件の埋め込みを加算し、シーケンス長方向にブロードキャストできるように次元を追加
        # emb.shape: (B, hidden_dim, 1)
        emb = (t_emb + c_emb).unsqueeze(-1)
        
        # メイン処理 (CNN)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, emb)
            
        out = self.output_proj(h)
        
        # 出力を元の (Batch, Length, Channel) に戻す
        # out.shape: (B, 2, 128) -> (B, 128, 2)
        return out.transpose(1, 2)