import torch
import torch.nn as nn
import torch.nn.functional as F


class Emb2Rep(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size,num_heads=8):
        super(Emb2Rep, self).__init__()
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1,bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 注意这里的大小是lstm_hidden_size的两倍，因为有两个方向
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        # attn_output, _ = self.multihead_attention(x, x, x)
        attn_output = x
        lstm_output, (h_n, c_n) = self.lstm(attn_output)
        # 双向LSTM的h_n是一个包含两个方向最后隐藏状态的张量，所以我们需要将它们合并起来
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        final_output = self.mlp(h_n)
        return final_output


def test():
    # 示例参数
    # Parameters for the model
    input_dim = 768  # Example input dimension
    hidden_dim = 512  # Example hidden dimension
    num_heads = 8  # Example number of heads in multihead attention
    output_size = 128
    # 创建模型
    model = Emb2Rep(input_dim=input_dim, hidden_dim=hidden_dim, output_size=output_size,
                    num_heads=num_heads)

    print(model)
    # 示例输入数据
    x = torch.rand(10, 50, input_dim)  # 假设有10个样本，每个样本是50个词，每个词的嵌入向量大小是256

    # 运行模型
    rep = model(x)
    print(rep.shape)  # 输出表示的形状

if __name__ == '__main__':
    test()
