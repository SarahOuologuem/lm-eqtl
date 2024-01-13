import torch
from torch import nn

from model.models.spec_dss import L1Block, DSS

class DenseRegressionHead(nn.Module): 

    def __init__(self, input_size, hidden_size):
        super(DenseRegressionHead, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ELU(), 
            nn.Linear(128, 64), 
            nn.ELU(), 
            nn.Linear(64, 32), 
            nn.ELU(), 
            nn.Linear(32, 1)
        )

    def forward(self, x): 
        x = self.model(x)
        return x


class RNNRegressionHead(nn.Module): 

    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressionHead, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x): 
        _, (hn, _) = self.rnn(x)
        output = self.linear(hn[-1, :, :])
        return output
    

class TransformerRegressionHead(nn.Module): 

    def __init__(self, input_size, num_layers=1):
        super(TransformerRegressionHead, self).__init__()

        self.transformer = nn.TransformerEncoderLayer(input_size, num_layers)
        self.linear = nn.Linear(32, 1)

    def forward(self, x): 
        x = self.transformer(x)
        print("SHAPE AFTER TRANSFORMER: ", x.size())
        x = x.reshape(-1, 32)
        print("SHAPE AFTER RESHAPE: ", x.size())
        output = self.linear(x)
        return output
    

class RegDSSResNet(nn.Module): 
    """DSSResNet with variable regression head"""

    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=128, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        species_encoder = None,
        embed_before = False, 
        regression_head=None
    ):
        super().__init__()

        self.prenorm = prenorm
        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((15-1)/2),kernel_size=15,bias=True)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                DSS(
                    d_model=d_model, 
                    l_max=1, 
                    bidirectional=True,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((15-1)/2),kernel_size=15,bias=True)

        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for x in range(3)])
        
        self.regression_head = regression_head

        self.species_encoder = species_encoder
        self.embed_before = embed_before

    def forward(self, x, xs):
        """
        Input x is shape (B, d_input, L)
        """
        u = x
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        if self.embed_before:
            x = self.species_encoder(x,xs)

        #print("shape after encoder", x.size())
        x = self.resnet_layer(x)


        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

                
        if not self.embed_before:
            x = self.species_encoder(x,xs)

        seq_embedding = x

        print("EMB BEFORE AGG: ", seq_embedding.size())

        reg_y = self.regression_head(seq_embedding)

        x = self.decoder(x)  # (B, d_model, L) -> (B, d_output, L)

        embeddings = {}
        embeddings["seq_embedding"] = seq_embedding

        # reconstructed sequence, embeddings and predicted expression for sequence
        return x, embeddings, reg_y
        
    


class DenseRegDSSResNet(nn.Module):
    """
    MLM  model with added regression head for multitask learning. 
    """
    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=128, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        species_encoder = None,
        embed_before = False,
        #embed='label'
    ):
        super().__init__()

        self.prenorm = prenorm
        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((15-1)/2),kernel_size=15,bias=True)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                DSS(
                    d_model=d_model, 
                    l_max=1, 
                    bidirectional=True,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((15-1)/2),kernel_size=15,bias=True)

        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for x in range(3)])

        self.regression_head = DenseRegressionHead(
            input_size=d_model, 
            hidden_size=256
        )

        self.species_encoder = species_encoder
        self.embed_before = embed_before


    def forward(self, x, xs):
        """
        Input x is shape (B, d_input, L)
        """
        u = x
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        if self.embed_before:
            x = self.species_encoder(x,xs)

        #print("shape after encoder", x.size())
        x = self.resnet_layer(x)


        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

                
        if not self.embed_before:
            x = self.species_encoder(x,xs)

        seq_embedding = x

        print("EMB BEFORE AGG: ", seq_embedding.size())

        # sum the embeddings across the sequence length dimension
        aggregated_embs = seq_embedding.mean(dim=-1)
        reg_y = self.regression_head(aggregated_embs)

        print("EMB AFTER AGG: ", seq_embedding.size())

        x = self.decoder(x)  # (B, d_model, L) -> (B, d_output, L)

        embeddings = {}
        embeddings["seq_embedding"] = seq_embedding

        # reconstructed sequence, embeddings and predicted expression for sequence
        return x, embeddings, reg_y
    