from __future__ import annotations
import argparse, yaml, torch, torch.nn as nn, time, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from torch_geometric.loader import NeighborLoader
from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import set_seed, Timer, compute_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, choices=["mlp", "gcn", "sage"], required=True)
    return p.parse_args()

def build_model(name, cfg, num_features, num_classes, device):
    if name == "mlp":
        return MLP(num_features, int(cfg["mlp"]["hidden_dim"]), num_classes, float(cfg["mlp"]["dropout"])).to(device)
    if name == "gcn":
        return GCN(num_features, int(cfg["gcn"]["hidden_dim"]), num_classes, float(cfg["gcn"]["dropout"])).to(device)
    return GraphSAGE(num_features, int(cfg["sage"]["hidden_dim"]), num_classes, float(cfg["sage"]["dropout"])).to(device)

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    data = load_cora()
    pyg_data = data.pyg_data.to(device)
    x, y = pyg_data.x, pyg_data.y
    edge_index = pyg_data.edge_index
    train_mask, val_mask, test_mask = pyg_data.train_mask, pyg_data.val_mask, pyg_data.test_mask

    model = build_model(args.model, cfg, data.num_features, data.num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    epochs = int(cfg["epochs"])
    print("device:", device, "| model:", args.model, "| epochs:", epochs)

    if args.model == "sage":
        bs = int(cfg["sampling"]["batch_size"])
        n1 = int(cfg["sampling"]["num_neighbors_l1"])
        n2 = int(cfg["sampling"]["num_neighbors_l2"])
        train_loader = NeighborLoader(
            pyg_data,
            input_nodes=train_mask,
            num_neighbors=[n1, n2],
            batch_size=bs,
            shuffle=True,
        )
    else:
        train_loader = None

    total_train_s = 0.0
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        if args.model in ["mlp", "gcn"]:
            with Timer() as t:
                logits = model(x) if args.model == "mlp" else model(x, edge_index)
                loss = criterion(logits[train_mask], y[train_mask])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_s += t.elapsed_s
        else:
            with Timer() as t:
                total_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    seed_size = int(batch.batch_size)
                    loss = criterion(out[:seed_size], batch.y[:seed_size])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total_loss += float(loss.item())
            total_train_s += t.elapsed_s
            loss = torch.tensor(total_loss / max(1, len(train_loader)))

        model.eval()
        with torch.no_grad():
            logits = model(x) if args.model == "mlp" else model(x, edge_index)
            m_train = compute_metrics(logits[train_mask], y[train_mask], data.num_classes)
            m_val   = compute_metrics(logits[val_mask],   y[val_mask],   data.num_classes)
            m_test  = compute_metrics(logits[test_mask],  y[test_mask],  data.num_classes)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} "
                  f"train_acc={m_train['acc']:.4f} val_acc={m_val['acc']:.4f} test_acc={m_test['acc']:.4f} "
                  f"train_f1={m_train['macro_f1']:.4f} val_f1={m_val['macro_f1']:.4f} test_f1={m_test['macro_f1']:.4f} "
                  f"epoch_time_s={t.elapsed_s:.4f}")

    print(f"total_train_time_s={total_train_s:.4f}")
    print(f"train_loop_time={time.time() - train_start:.4f}")

    # Checkpoint
    os.makedirs("TP4/runs", exist_ok=True)
    ckpt_path = os.path.join("TP4/runs", f"{args.model}.pt")
    torch.save({"model": args.model, "config_path": args.config, "state_dict": model.state_dict()}, ckpt_path)
    print("checkpoint_saved:", ckpt_path)

if __name__ == "__main__":
    main()
