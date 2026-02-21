from __future__ import annotations
import argparse, yaml, torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import set_seed, Timer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, choices=["mlp", "gcn", "sage"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()

def build_model(name, cfg, num_features, num_classes):
    if name == "mlp":
        return MLP(num_features, int(cfg["mlp"]["hidden_dim"]), num_classes, float(cfg["mlp"]["dropout"]))
    if name == "gcn":
        return GCN(num_features, int(cfg["gcn"]["hidden_dim"]), num_classes, float(cfg["gcn"]["dropout"]))
    return GraphSAGE(num_features, int(cfg["sage"]["hidden_dim"]), num_classes, float(cfg["sage"]["dropout"]))

def sync_if_cuda(device):
    if device.type == "cuda": torch.cuda.synchronize()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_cora()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    model = build_model(args.model, cfg, data.num_features, data.num_classes).to(device)
    model.eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    warmup, runs = 10, 50

    def forward_once():
        if args.model == "mlp": return model(x)
        return model(x, edge_index)

    with torch.no_grad():
        for _ in range(warmup): forward_once()
        sync_if_cuda(device)
        elapsed = 0.0
        for _ in range(runs):
            sync_if_cuda(device)
            with Timer() as t: forward_once()
            sync_if_cuda(device)
            elapsed += t.elapsed_s

    avg_ms = 1000.0 * elapsed / runs
    print("model:", args.model)
    print("device:", device)
    print("avg_forward_ms:", round(avg_ms, 4))
    print("num_nodes:", int(x.shape[0]))
    print("ms_per_node_approx:", round(avg_ms / float(x.shape[0]), 8))

if __name__ == "__main__":
    main()
