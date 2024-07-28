import torch
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os

def self_looped_adj_mat(adj_mat):
    adj_mat_out_self = adj_mat +torch.eye(adj_mat.size(0))
    degree_mat_self = torch.diag(torch.sqrt(torch.reciprocal(torch.sum(adj_mat_out_self,dim=0))))
    adj_mat_out = torch.mm(degree_mat_self, torch.mm(adj_mat_out_self, degree_mat_self))
    return adj_mat_out


def get_dataset(args, transform = 'SSAMSE'):
    if args.mesh_type=='square_grid':
        path = "./dataset/dataset_square_grid.mat"

    elif args.mesh_type=='square_mesh':
        path = "./dataset/dataset_square_mesh.mat"

    elif args.mesh_type=='face':
        path = "./dataset/dataset_face_grid_elec_5e9.mat"

    elif args.mesh_type =='cylinder':
        path = "./dataset/dataset_cylinder.mat"

    elif args.mesh_type =='curvilinear':
        path = "./dataset/dataset_curvilinear.mat"

    elif args.mesh_type =='hand':
        path = "./dataset/dataset_hand.mat"

    elif args.mesh_type == 'cylinder':
        path = "./dataset/dataset_cylinder.mat"

    Dataset = h5py.File(path,'r')
    dataset = Dataset['dataset']
    dataset_input = torch.FloatTensor(dataset['/dataset/input_data'][()]).permute(1, 0)
    dataset_output_raw = torch.FloatTensor(dataset['/dataset/output_data'][()]).permute(1, 0)
    adj_mat_out = torch.FloatTensor(dataset['/dataset/adj_mat_out'][()])
    node_coord = torch.FloatTensor(dataset['/dataset/elem_ctr'][()]).permute(1, 0)
    volume_weight = torch.FloatTensor(dataset['/dataset/elem_vol'][()]).permute(1, 0)

    dataset_output = dataset_output_raw

    # Boundary extraction
    if args.mesh_type == 'face':
        bd_elem_info = np.loadtxt("./dataset/face_elems_boundary.csv", delimiter=",")
        dataset_output = dataset_output[:, bd_elem_info]
        dataset_output_raw = dataset_output_raw[:, bd_elem_info]
        adj_mat_out = adj_mat_out[bd_elem_info][:, bd_elem_info]
        node_coord = node_coord[bd_elem_info]
        volume_weight = volume_weight[bd_elem_info]

    volume_weight = volume_weight / torch.sum(volume_weight)

    # Data split
    max_data_size = dataset_input.size(0)
    input_size = dataset_input.size(1)
    output_size = dataset_output.size(1)

    dataset_idx = int(max_data_size * args.dataset_ratio)
    split_idx_trn = int(dataset_idx * args.trn_ratio)
    split_idx_val = int(dataset_idx * (args.trn_ratio + args.val_ratio))

    dataset_input_trn = dataset_input[:split_idx_trn, :]
    dataset_input_val = dataset_input[(split_idx_trn + 1):split_idx_val, :]
    dataset_input_tst = dataset_input[(split_idx_val + 1):, :]

    dataset_output_trn = dataset_output[:split_idx_trn, :]
    dataset_output_val = dataset_output[(split_idx_trn + 1):split_idx_val, :]
    dataset_output_tst = dataset_output[(split_idx_val + 1):, :]
    dataset_output_tst_raw = dataset_output_raw[(split_idx_val + 1):, :]

    print("Dataset processing done")
    del dataset_input, dataset_output, dataset_output_raw

    dataset_trn = EITGNNDataset(dataset_input_trn, dataset_output_trn)
    dataset_val = EITGNNDataset(dataset_input_val, dataset_output_val)
    dataset_tst = EITGNNDataset(dataset_input_tst, dataset_output_tst)
    dataset_tst_raw = EITGNNDataset(dataset_input_tst, dataset_output_tst_raw)

    cfg = {
        'input_size': input_size,
        'output_size': output_size,
        'adj_mat_out': adj_mat_out,
        'node_coord': node_coord,
        'volume_weight': volume_weight
    }

    if args.model_type == 'OGN_GCN':
        if args.mesh_type == 'face':
            recon_mat = torch.FloatTensor(np.loadtxt("recon_mat_face.csv", delimiter=",")).permute(1, 0)
            recon_mat = recon_mat[:, bd_elem_info]
        elif args.mesh_type == 'square_grid':
            recon_mat = torch.FloatTensor(np.loadtxt("recon_mat_square_grid.csv", delimiter=",")).permute(1, 0)
        elif args.mesh_type == 'square_mesh':
            recon_mat = torch.FloatTensor(np.loadtxt("recon_mat_square_mesh.csv", delimiter=",")).permute(1, 0)
            cfg['recon_mat'] = recon_mat
        elif args.mesh_type == 'curvilinear':
            recon_mat = torch.FloatTensor(np.loadtxt("recon_mat_curvilinear.csv", delimiter=",")).permute(1, 0)
            cfg['recon_mat'] = recon_mat
        elif args.mesh_type == 'cylinder':
            recon_mat = torch.FloatTensor(np.loadtxt("recon_mat_cylinder.csv", delimiter=",")).permute(1, 0)
        cfg['recon_mat'] = recon_mat

    return dataset_trn, dataset_val, dataset_tst, dataset_tst_raw, cfg


class EITGNNDataset(Dataset):
    def __init__(self, dataset_input, dataset_output):
        super(EITGNNDataset, self).__init__()
        self.input = dataset_input
        self.output = dataset_output

    def process(self):
        self.input = self.input

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, adj_mat):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights.requires_grad_(False)
        self.lap_mat = torch.diag(torch.sum(adj_mat, dim=1)) - adj_mat

    def forward(self, input, target):
        squared_error = (input - target) ** 2
        reg_error = (torch.matmul(input, self.lap_mat))**2
        loss_recon = torch.mean(torch.sum(squared_error * self.weights.t(), 1))
        loss_lap = torch.mean(torch.sum(reg_error * self.weights.t(), 1))
        return loss_recon, loss_lap


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.early_stop = False
        self.loss_history = []

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model
        self.loss_history.append(val_loss)
