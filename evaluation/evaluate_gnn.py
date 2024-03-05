from model.gnn import *
from config import *



def train_gnn():
    model = model.to(device)
    loss_values = []
    epochs = range(1, 15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in epochs:
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            # print(f"The sampled data is: {sampled_data}")
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["paper", "has", "label"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        # Calculate average loss for the epoch
        avg_loss = total_loss / total_examples
        loss_values.append(avg_loss)  # Append to the list
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    # Create a simple plot
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epochs")
    plt.grid(True)
    plt.show()
