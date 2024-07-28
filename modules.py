import math

from einops import parse_shape, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

def add_noise(noise, input):
    device = input.device
    noise_val = (torch.randn(input.size())*noise).to(device)
    output = input + noise_val
    return output

# https://github.com/pytorch/pytorch/issues/14489
def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)


# sparse-to-dense elementwise-multiplication to sparse tensor
def sdem2s(sparse_tensor, dense_tensor):
    # Ensure that the sparse tensor has the same device as the dense tensor
    # Perform element-wise multiplication without converting to dense
    result_sparse = torch.sparse_coo_tensor(
        indices=sparse_tensor._indices(),
        values=sparse_tensor._values() * dense_tensor[sparse_tensor._indices()[0], sparse_tensor._indices()[1]],
        size=sparse_tensor.size()
    )
    return result_sparse


# sparse-to-sparse element-wise multiplication to sparse tensor
# the second sparse involves only the values, as the two intended values share common indices.
def ssem2s(sparse_tensor, dense_tensor):
    # Ensure that the sparse tensor has the same device as the dense tensor
    # Perform element-wise multiplication without converting to dense
    result_sparse = torch.sparse_coo_tensor(
        indices=sparse_tensor._indices(),
        values=sparse_tensor._values() * dense_tensor,
        size=sparse_tensor.size()
    )
    return result_sparse


# sparse-to-batched dense tensor matrix multiplication
def sbdmm(sparse_tensor, batched_dense_tensor):
    bsz, isz, fsz = batched_dense_tensor.size()
    result_batched_dense = torch.sparse.mm(sparse_tensor, batched_dense_tensor.permute(1, 0, 2).reshape(isz, -1)).reshape(isz, bsz, fsz).permute(1, 0,
                                                                                                                 2)
    return result_batched_dense


class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, A, node_coord=None):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.fc = nn.Linear(in_features, out_features)
        self.sparse = True

        self.A = self.self_looped_adj_mat(A)

        # temp for sparse
        if self.sparse:
            self.A = self.A.to_sparse()

    def self_looped_adj_mat(self, adj_mat):

        adj_mat_out_self = adj_mat + torch.eye(adj_mat.size(0)).to(adj_mat.device)
        degree_mat_self = torch.diag(torch.sqrt(torch.reciprocal(torch.sum(adj_mat_out_self, dim=0))))
        adj_mat_out = torch.mm(degree_mat_self, torch.mm(adj_mat_out_self, degree_mat_self))
        return adj_mat_out.to(adj_mat.device)

    def forward(self, x):
        device = torch.get_device(x)
        if self.sparse:
            out = sbdmm(self.A.to(device), x)
        else:
            out = batch_mm(self.A, x)

        out = self.fc(out)
        return out


class GraphNorm(nn.Module):
    def __init__(self, feature_num, eps=1e-5):
        super(GraphNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_num))  # Learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(feature_num))  # Learnable shift parameter
        self.alpha = nn.Parameter(torch.ones(feature_num))
        # given input of B x N x F
        # Learnable parameters are: weight (F), and bias(F), and mean scaler, as alpha (F)
        # given input, the mean is computed on N, which thereby results in the value set of B x F.
        # Then, std is computed by (input - alpha * mean), resulting in the value set of B x F as well.
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        scaled_x = x - mean * self.alpha.view(1, 1, -1)
        var = scaled_x.var(dim=1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma.view(1, 1, -1) * x_normalized + self.beta.view(1, 1, -1)
        return out



