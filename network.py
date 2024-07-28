import torch
import torch.nn as nn
from modules import add_noise, GCN_layer, SAGE_layer, GIN_layer, GraphNorm


def ModelGen(cfg, args):
    if args.model_type == 'MLP_MLP':
        return MLP_MLP(cfg, args)

    elif args.model_type == 'TF_MLP':
        return TF_MLP(cfg, args)

    elif args.model_type == 'MLP_CNN':
        return MLP_CNN(cfg, args)

    elif args.model_type == 'TF_CNN':
        return TF_CNN(cfg, args)

    elif args.model_type == 'MLP_GCN':
        return MLP_GCN(cfg, args)

    elif args.model_type == 'OGN_GCN':
        return OGN_GCN(cfg,args)

    elif args.model_type == 'TF_GCN':
        return TF_GCN(cfg, args)

    elif args.model_type == 'TF_GIN':
        return TF_GIN(cfg, args)

    elif args.model_type == 'TF_SAGE':
        return TF_SAGE(cfg, args)


class MLP_MLP(nn.Module):
    def __init__(self, cfg, args):
        super(MLP_MLP, self).__init__()
        # Sensor encoder
        self.L_FC_1 = nn.Linear(cfg['input_size'], cfg['input_size'])
        self.L_FC_2 = nn.Linear(cfg['input_size'], cfg['input_size'])
        self.L_FC_3 = nn.Linear(cfg['input_size'], cfg['input_size'])
        self.L_BN_1 = nn.BatchNorm1d(cfg['input_size'])
        self.L_BN_2 = nn.BatchNorm1d(cfg['input_size'])
        self.L_BN_3 = nn.BatchNorm1d(cfg['input_size'])

        # Image decoder
        self.L_FC_4 = nn.Linear(cfg['input_size'], cfg['output_size'])
        self.L_FC_5 = nn.Linear(cfg['output_size'], cfg['output_size'])

        self.L_BN_4 = nn.BatchNorm1d(cfg['output_size'])

        # Utils
        self.L_ReLU = nn.ReLU()
        self.noise = args.noise

    def forward(self, x):
        out = add_noise(self.noise, x)

        # Sensor encoder
        out = self.L_FC_1(out)
        out = self.L_BN_1(out)
        out = self.L_ReLU(out)
        #
        out = self.L_FC_2(out)
        out = self.L_BN_2(out)
        out = self.L_ReLU(out)

        out = self.L_FC_3(out)
        out = self.L_BN_3(out)
        out = self.L_ReLU(out)

        # Image decoder
        out = self.L_FC_4(out)
        out = self.L_BN_4(out)
        out = self.L_ReLU(out)

        out = self.L_FC_5(out)

        return out


class TF_CNN(nn.Module):
    def __init__(self, cfg, args):
        super(TF_CNN, self).__init__()
        # Sensor encoder
        self.patch_length = cfg['input_size']
        self.query_dim = 1
        self.feature_amp = args.feature_num

        self.patch_embedding = nn.Parameter(torch.empty(self.patch_length, self.feature_amp))
        nn.init.xavier_normal_(self.patch_embedding)

        self.positional_embedding = self.positional_encoding()

        self.TF_enc = nn.TransformerEncoderLayer(d_model=self.feature_amp, nhead=8, dim_feedforward=64,
                                                 dropout=0, batch_first=True)
        self.L_TF = nn.TransformerEncoder(self.TF_enc, num_layers=args.TF_layers)

        # Domain transform
        self.domain_transform = nn.Parameter(torch.empty((cfg['input_size'], cfg['output_size'])))
        nn.init.xavier_normal_(self.domain_transform)

        # image decoder
        self.L_Conv_1 = nn.Conv2d(self.feature_amp, 32, kernel_size=3, padding=1, bias=False)
        self.L_Conv_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.L_TrConv_1 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, bias=False)
        self.L_ConvBN_1 = nn.BatchNorm2d(32)
        self.L_ConvBN_2 = nn.BatchNorm2d(32)

        # Utils
        self.L_ReLU = nn.ReLU()
        self.L_sigmoid = nn.Sigmoid()
        self.noise = args.noise

    def positional_encoding(self):
        pe = torch.zeros(self.patch_length, self.feature_amp)
        pe.requires_grad = False
        pos = torch.arange(0, self.patch_length).unsqueeze(dim=1)
        _2i = torch.arange(0, self.feature_amp, step=2)

        pe[:, 0::2] = torch.sin(pos / 10000 ** (_2i / self.feature_amp))
        pe[:, 1::2] = torch.cos(pos / 10000 ** (_2i / self.feature_amp))

        return pe

    def forward(self, x):
        out = add_noise(self.noise, x)  # Input size : B x I / B : Batch size, I : Input size
        device = torch.get_device(x)
        # Sensor encoder
        out = out.unsqueeze(2) * self.patch_embedding.unsqueeze(0)
        out = out + self.positional_embedding.to(device)
        out = self.L_TF(out)
        out = out.view(-1, self.patch_length * self.query_dim,
                       self.feature_amp)  # Changing the feature into B x I (= P x Q) x F

        # Domain transform
        out = torch.einsum('b i j, i k -> b k j', out, self.domain_transform)  # into B x O x F

        # Imgage decoder
        out = out.permute(0, 2, 1).view(-1, self.feature_amp, 24, 24)
        out = self.L_Conv_1(out)
        out = self.L_ConvBN_1(out)
        out = self.L_ReLU(out)

        out = self.L_Conv_2(out)
        out = self.L_ConvBN_2(out)
        out = self.L_ReLU(out)

        out = self.L_TrConv_1(out)
        out = out.view(-1, 576)

        return out


class OGN_GCN(nn.Module):
    def __init__(self, cfg, args):
        super(OGN_GCN, self).__init__()
        # Sensor encoder

        self.L_OGN = cfg['recon_mat'].requires_grad_(False).cuda()
        # nn.init.xavier_normal_(self.L_FC_DT_1)

        # Image decoder
        hidden_dim_GNN = args.feature_num
        self.node_coord = cfg['node_coord']
        self.att_switch = args.pos_att_switch
        self.adj_mat_out = cfg['adj_mat_out']

        self.dec_layer_num = 5
        for i in range(1, self.dec_layer_num + 1):
            if i == 1:
                setattr(self, f'L_GNN_dec_{i}',
                        GCN_layer(1, hidden_dim_GNN, self.adj_mat_out, node_coord=self.node_coord,
                                  att_switch=self.att_switch))
            else:
                setattr(self, f'L_GNN_dec_{i}',
                        GCN_layer(hidden_dim_GNN, hidden_dim_GNN, self.adj_mat_out, node_coord=self.node_coord,
                                  att_switch=self.att_switch))
            setattr(self, f'L_N_dec_{i}', GraphNorm(hidden_dim_GNN))

        self.L_FC_readout = nn.Sequential(
            nn.Linear(hidden_dim_GNN, hidden_dim_GNN),
            GraphNorm(hidden_dim_GNN),
            nn.ReLU(),
            nn.Linear(hidden_dim_GNN, hidden_dim_GNN),
            GraphNorm(hidden_dim_GNN),
            nn.ReLU(),
            nn.Linear(hidden_dim_GNN, 1)
        )
        # Utils
        self.L_ReLU = nn.ReLU()
        self.L_sigmoid = nn.Sigmoid()
        self.noise = args.noise

    def forward(self, x):
        out = add_noise(self.noise, x) # Input size : B x I / B : Batch size, I : Input size
        # Sensor encoder
        out = torch.matmul(out, self.L_OGN)
        out = out.unsqueeze(2)

        for i in range(1, self.dec_layer_num + 1):
            out = self.L_ReLU(getattr(self, f'L_N_dec_{i}')(getattr(self, f'L_GNN_dec_{i}')(out))) + out
        out = self.L_FC_readout(out) # Do not perform aggregation at the last layer
        out = out.squeeze(2) # Out size : B x O / B : Batch size, O : Output size

        return out


class TF_GCN(nn.Module):
    def __init__(self, cfg, args):
        super(TF_GCN, self).__init__()
        # Sensor encoder
        self.patch_length = cfg['input_size']
        self.feature_amp = args.feature_num
        self.patch_embedding = nn.Parameter(torch.empty(self.patch_length, self.feature_amp))
        nn.init.xavier_normal_(self.patch_embedding)
        self.positional_embedding = self.positional_encoding()
        self.TF_enc = nn.TransformerEncoderLayer(d_model=self.feature_amp, nhead=8, dim_feedforward=64,
                                                 dropout=0, batch_first=True)
        self.L_TF = nn.TransformerEncoder(self.TF_enc, num_layers=args.TF_layers)

        # Domain transform
        self.L_FC_DT_1 = nn.Parameter(torch.empty((cfg['input_size'], cfg['output_size'])))
        nn.init.xavier_normal_(self.L_FC_DT_1)

        # Image decoder
        self.hidden_dim_GNN = args.feature_num
        self.node_coord = cfg['node_coord'].cuda()
        self.att_switch = args.pos_att_switch
        self.adj_mat_out = cfg['adj_mat_out'].cuda()
        if self.feature_amp != self.hidden_dim_GNN:
            self.L_FC_DT_2 = nn.Linear(self.feature_amp, self.hidden_dim_GNN, bias=False)

        self.dec_layer_num = args.GCN_layers
        for i in range(1, self.dec_layer_num + 1):
            setattr(self, f'L_GNN_dec_{i}',
                    GCN_layer(self.hidden_dim_GNN, self.hidden_dim_GNN, self.adj_mat_out, node_coord=self.node_coord,
                              att_switch=self.att_switch))
            setattr(self, f'L_N_dec_{i}', GraphNorm(self.hidden_dim_GNN))

        self.readout = args.readout

        if self.readout == 'MLP':
            self.L_FC_readout = nn.Sequential(
                nn.Linear(self.hidden_dim_GNN, self.hidden_dim_GNN),
                GraphNorm(self.hidden_dim_GNN),
                nn.ReLU(),
                nn.Linear(self.hidden_dim_GNN, self.hidden_dim_GNN),
                GraphNorm(self.hidden_dim_GNN),
                nn.ReLU(),
                nn.Linear(self.hidden_dim_GNN, 1)
            )
        elif self.readout == 'Linear':
            self.L_FC_readout = nn.Linear(self.hidden_dim_GNN, 1)

        # Utils
        self.L_ReLU = nn.ReLU()
        self.L_sigmoid = nn.Sigmoid()
        self.noise = args.noise

    def positional_encoding(self):
        pe = torch.zeros(self.patch_length, self.feature_amp)
        pe.requires_grad = False
        pos = torch.arange(0, self.patch_length).unsqueeze(dim=1)
        _2i = torch.arange(0, self.feature_amp, step=2)

        pe[:, 0::2] = torch.sin(pos / 10000 ** (_2i / self.feature_amp))
        pe[:, 1::2] = torch.cos(pos / 10000 ** (_2i / self.feature_amp))

        return pe

    def forward(self, x):
        out = add_noise(self.noise, x)  # Input size : B x I / B : Batch size, I : Input size
        device = torch.get_device(x)
        out = out.unsqueeze(2) * self.patch_embedding.unsqueeze(0)
        out = out + self.positional_embedding.to(device)
        out = self.L_TF(out)
        out = out.view(-1, self.patch_length * self.query_dim,
                       self.feature_amp)  # Changing the feature into B x I (= P x Q) x F

        # Domain transform
        out = torch.matmul(out.permute(0, 2, 1), self.L_FC_DT_1).permute(0, 2, 1)
        if self.feature_amp != self.hidden_dim_GNN:
            out = self.L_FC_DT_2(out)
            out = self.L_ReLU(out)

        for i in range(1, self.dec_layer_num + 1):
            out = self.L_ReLU(getattr(self, f'L_N_dec_{i}')(getattr(self, f'L_GNN_dec_{i}')(out))) + out

        if self.readout == 'MLP' or self.readout == 'Linear':
            out = self.L_FC_readout(out)  # Do not perform aggregation at the last layer
        elif self.readout == 'Avg':
            out = torch.mean(out, dim=-1, keepdim=True)
        elif self.readout == 'Max':
            out, _ = torch.max(out, dim=-1, keepdim=True)

        out = out.squeeze(2)  # Out size : B x O / B : Batch size, O : Output size

        return out
