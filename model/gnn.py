class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_paper: Tensor, x_label: Tensor, edge_label_index: Tensor) -> Tensor:
        # print(f"The edge label index is: {edge_label_index}")
        # Convert node embeddings to edge-level representations:
        edge_feat_label = x_label[edge_label_index[0]]
        edge_feat_paper = x_paper[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_paper * edge_feat_label).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.paper_lin = torch.nn.Linear(209, hidden_channels)
        self.label_emb = torch.nn.Embedding(data["label"].num_nodes, hidden_channels)
        self.paper_emb = torch.nn.Embedding(data["paper"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "label": self.label_emb(data["label"].node_id),
            "paper": self.paper_lin(data["paper"].x) + self.paper_emb(data["paper"].node_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["label"],
            x_dict["paper"],
            data["paper", "has", "label"].edge_label_index,
        )
        return pred


if __name__ == "__main__":
    model = Model(hidden_channels=16)