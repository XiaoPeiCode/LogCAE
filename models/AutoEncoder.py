from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Encoder, self).__init__()
        layer_dims = [input_dim] + hidden_dims
        self.layers = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]), nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]), nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]),  nn.ReLU())

    def forward(self, input_data):
        hidden = self.layers(input_data)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dims):
        super(Decoder, self).__init__()
        layer_dims = hidden_dims + [output_dim]
        self.layers = nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]),  nn.ReLU(),
            nn.Linear(layer_dims[1], layer_dims[2]),  nn.ReLU(),
            nn.Linear(layer_dims[2], layer_dims[3]),nn.Tanh()
        )


    def forward(self, input_data):
        output = self.layers(input_data)
        return output

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(AE, self).__init__()
        self.hidden_dims = hidden_dims
        self.criterion = nn.MSELoss(reduction="none")
        self.input_dim = input_dim
        # self.embedder = Embedder(vocab_size, embedding_dim)
        # self.bertembedder = BERTEmbedder()
        #
        # self.rnn = nn.LSTM(
        #     input_size=embedding_dim,
        #     hidden_size=self.hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     # bidirectional=(self.num_directions == 2),
        # )
        self.encoder = Encoder(input_dim, hidden_dims)
        # self.clustering_layer = nn.Linear(hidden_dims[-1], num_clusters)
        # Use BERTEmbedder here
        self.decoder = Decoder(input_dim, list(reversed(hidden_dims)))



    def forward(self, input_data):


        # 都不用只用AE
        representation = input_data
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)

        # 假设 representation 可能是二维或三维
        if representation.dim() == 3:
            # 如果 representation 是三维，按照最后两个维度求平均
            pred = self.criterion(representation, decoded).mean(dim=(-1, -2))
        elif representation.dim() == 2:
            # 如果 representation 是二维，按照最后一个维度求平均
            pred = self.criterion(representation, decoded).mean(dim=-1)

        # pred = self.criterion(representation, decoded).mean(dim=(-1,-2))
        # pred = self.criterion(representation, representation).mean(dim=-1)
        # pred should be (n_sample),loss,should be (1)

        loss = pred.mean()
        # loss = pred.mean()
        return_dict = {"loss": loss,
                       "y_pred": pred,
                       "encoded":encoded,
                       "rep":encoded}
        return return_dict

