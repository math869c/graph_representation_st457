from keras.layers import *
import torch.nn as nn
import torch
import torch.nn.functional as F

# order of blocks
# 1. LSTM
# 2. TGC
# 3. GAT

# LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_units, output_size, dropout_rate):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(
            input_size=hidden_units,
            hidden_size=hidden_units * 2,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_units * 2, hidden_units)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x




# The TGC consists of 3 building blocks
# Block 1: The LSTM Encoder
class CompanyEncoder(nn.Module):
    def __init__(self, num_features, emb_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=emb_dim, batch_first=True)

    def forward(self, x):
        # x: [T, L, N, F]
        # Letter for dimensions: T is time periods, L length of sequences, N number of firms, F number of features, B batch size
        B, L, N, F = x.shape

        x = x.permute(0, 2, 1, 3).reshape(B * N, L, F)   # [B*N, L, F]

        _, (h_n, _) = self.lstm(x)
        E = h_n[-1].reshape(B, N, -1)                    # [B, N, emb_dim]
        return E

# Block 2: The Graph Layer
class TemporalMultiRelationGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.edge_linear = nn.Linear(num_relations, 1)

    def forward(self, E, A):
        # E: [B, N, d]
        # A: [N, N, K]
        # setiup
        B, N, d = E.shape
        E_E = torch.matmul(E, E.transpose(1, 2)) # Same as e_i^T e_j in paper

        edge_score = self.edge_linear(A).squeeze(-1)
        edge_score = torch.relu(edge_score) # this is phi in the paper

        deg = edge_score.sum(dim=0, keepdim=True) + 1e-6
        edge_weight = edge_score / deg                 # phi/d

        weights = E_E * edge_weight.unsqueeze(0)       # Combine: part_1 = e_i^T e_j @ phi/d
        H = torch.matmul(weights, E)                   # combine: part_1@ e_j

        return H

# Block 3: The Dense Layer
# I think I did one extra ReLu/linear layer too much
class Header(nn.Module):
  def __init__(self, gcn_dim):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(gcn_dim, 64),
        nn.ReLU(), # I Think this should be commented out
        nn.Dropout(0.2),  # I Think this should be commented out
        nn.Linear(64, 1)  # I Think this should be commented out
    )

  def forward(self, x):
    return self.model(x)

class TGCModel(nn.Module):
    def __init__(self, num_features, emb_dim, gcn_dim, num_relations):
        super().__init__()
        self.encoder = CompanyEncoder(num_features, emb_dim)
        self.tgc = TemporalMultiRelationGCN(emb_dim, gcn_dim, num_relations)
        self.head = Header(gcn_dim)

    def forward(self, x, A):
        # x: [B, L, N, F]
        # A: [N, N, K]
        E = self.encoder(x)               # Embedding layer
        H = self.tgc(E, A)                # combines embeddings and adjacency matrix
        y_hat = self.head(H).squeeze(-1)  # makes N predictions
        return y_hat

# GAT model
class GATLayer(nn.Module):
    """
    From seminar 7 with minor adjustments 
    """

    def __init__(self, c_in, c_out, num_relations, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.num_relations = num_relations

        if self.concat_heads:
            assert (
                c_out % num_heads == 0
            ), "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projections = nn.ModuleList([nn.Linear(c_in, c_out * num_heads, bias=False) for _ in range(num_relations)])
        self.a = nn.Parameter(torch.Tensor(num_relations, num_heads, 2 * c_out))
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        R = self.num_relations
        if adj_matrix.dim() == 3:
            adj_matrix = adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # We have only changed this little part to include multiple relations
        rel_outputs = []
        node_feats_input = node_feats

        for r in range(R):
            adj_r = adj_matrix[:,:,:, r]

            # Apply linear layer and sort nodes by head
            node_feats_r = self.projections[r](node_feats_input)

            node_feats_r = node_feats_r.view(batch_size, num_nodes, self.num_heads, -1)

            # We need to calculate the attention logits for every edge in the adjacency matrix
            # Doing this on all possible combinations of nodes is very expensive
            # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
            edges = adj_r.nonzero(as_tuple=False)  # Returns indices where the adjacency matrix is not 0 => edges

            node_feats_flat = node_feats_r.view(batch_size * num_nodes, self.num_heads, -1)
            edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
            edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
            a_input = torch.cat(
                [
                    torch.index_select(
                        input=node_feats_flat, index=edge_indices_row, dim=0
                    ),
                    torch.index_select(
                        input=node_feats_flat, index=edge_indices_col, dim=0
                    ),
                ],
                dim=-1,
            )  # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

            # Calculate attention MLP output (independent for each head)
            attn_logits = torch.einsum("ehc,hc->eh", a_input, self.a[r])
            attn_logits = self.leakyrelu(attn_logits)

            # Map list of attention values back into a matrix
            attn_matrix = attn_logits.new_full((batch_size, num_nodes, num_nodes, self.num_heads), -9e15)
            mask = adj_r != 0
            attn_matrix[mask[..., None].repeat(1, 1, 1, self.num_heads)] = attn_logits.reshape(-1)

            # Weighted average of attention
            attn_probs = nn.functional.softmax(attn_matrix, dim=2)

            if print_attn_probs:
                print(
                    "Attention probs\n", attn_probs.permute(0, 3, 1, 2)
                )  # show the attention weight for each edge
            out_r = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats_r)
            rel_outputs.append(out_r)

        # sum relation-specific messages
        node_feats = torch.stack(rel_outputs, dim=0).sum(dim=0)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats
    
class GAT(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_relations, num_heads=4, alpha=0.2):
        super().__init__()

        self.gat1 = GATLayer(
            c_in=c_in,
            c_out=c_hidden,
            num_relations = num_relations,
            num_heads=num_heads,
            concat_heads=True,
            alpha=alpha
        )

        self.gat2 = GATLayer(
            c_in=c_hidden,
            c_out=c_out,
            num_relations = num_relations,
            num_heads=1,
            concat_heads=False,
            alpha=alpha
        )

    def forward(self, x, adj):
        x = self.gat1(x, adj)      # [B, N, c_hidden]
        x = nn.functional.elu(x)
        x = self.gat2(x, adj)      # [B, N, c_out]
        x = x.squeeze(-1)
        return x
    
# RotatE model
class RotatE(nn.Module):
    def __init__(self, num_relations, emb_dim):
        super().__init__()

        self.emb_dim = emb_dim

        # relation phases (angles)
        self.phase = nn.Parameter(torch.randn(num_relations, emb_dim))

    def score_triples(self, z, edge_index, edge_type):
            """
            z: (B, N, 2*d)
            edge_index: (E, 2)
            edge_type: (E,)
            returns: (B, E)
            """
            B, N, D = z.shape
            d = D // 2
            assert D == 2 * self.emb_dim, f"Expected last dim {2*self.emb_dim}, got {D}"

            z_re = z[..., :d]
            z_im = z[..., d:]

            u = edge_index[:, 0]
            v = edge_index[:, 1]
            r = edge_type

            zu_re = z_re[:, u, :]
            zu_im = z_im[:, u, :]
            zv_re = z_re[:, v, :]
            zv_im = z_im[:, v, :]

            phase = self.phase[r]
            r_re = torch.cos(phase).unsqueeze(0)
            r_im = torch.sin(phase).unsqueeze(0)

            rot_re = zu_re * r_re - zu_im * r_im
            rot_im = zu_re * r_im + zu_im * r_re

            diff_re = rot_re - zv_re
            diff_im = rot_im - zv_im

            score = -(diff_re.pow(2) + diff_im.pow(2)).sum(dim=-1)
            return score

    def forward(self, z, edge_index, edge_type):
        return self.score_triples(z, edge_index, edge_type)
    
class GAT_RotatE(nn.Module):
    def __init__(self, c_in, c_hidden, emb_dim, num_relations, num_heads=4, alpha=0.2):
        super().__init__()
        self.encoder = GAT(
            c_in=c_in,
            c_hidden=c_hidden,
            c_out=2 * emb_dim,
            num_relations=num_relations,
            num_heads=num_heads,
            alpha=alpha
        )
        self.regressor = nn.Linear(2 * emb_dim, 1)
        self.rotate = RotatE(num_relations=num_relations, emb_dim=emb_dim)

    def forward(self, x, adj):
        z = self.encoder(x, adj)                 # (B, N, 2*emb_dim)
        z = self.rotate(z, adj)                  # relation transformation embedding      
        y_hat = self.regressor(z).squeeze(-1)    # (B, N)
        return y_hat, z