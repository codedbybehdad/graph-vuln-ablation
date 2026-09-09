"""Minimal CPU sanity check for the GGNN readout and gradients."""
import torch
from torch_geometric.data import Batch, Data
from train_ggnn import GGNN


def main():
    graphs = []
    for i, n in enumerate([20, 30, 25, 18]):
        edge_index = torch.randint(0, n, (2, n * 3), dtype=torch.long)
        edge_type = torch.randint(0, 3, (edge_index.size(1),), dtype=torch.long)
        graphs.append(Data(
            x=torch.randn(n, 138),
            edge_index=edge_index,
            edge_type=edge_type,
            y=torch.tensor([i % 2], dtype=torch.long),
        ))

    batch = Batch.from_data_list(graphs)
    model = GGNN(138, hidden_dim=200, num_steps=6, num_relations=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.3e-6)
    criterion = torch.nn.BCELoss()

    before = model.mlp_z.weight.detach().clone()
    for epoch in range(3):
        optimizer.zero_grad(set_to_none=True)
        probs = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(probs.clamp(1e-6, 1 - 1e-6), batch.y.float().view(-1))
        loss.backward()
        optimizer.step()
        grad_sum = sum(
            p.grad.detach().abs().sum().item()
            for p in model.parameters() if p.grad is not None
        )
        print(f"epoch={epoch + 1} loss={loss.item():.6f} prob_mean={probs.mean().item():.6f} grad_sum={grad_sum:.6e}")

    change = (model.mlp_z.weight.detach() - before).abs().sum().item()
    assert change > 0, "Model parameters did not change."
    assert torch.isfinite(probs).all(), "Non-finite model output."
    print(f"parameter_change={change:.6e}")
    print("SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
