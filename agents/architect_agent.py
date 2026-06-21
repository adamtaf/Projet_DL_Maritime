import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from agents.base_agent import BaseAgent

class HybridCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, conv_filters, hidden_dim, output_dim, ablation_mode=None):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, conv_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_filters if ablation_mode != "spatial" else embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim if ablation_mode != "temporal" else conv_filters, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        if self.ablation_mode == "spatial":
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1d(x))
        x = x.transpose(1, 2)
        if self.ablation_mode == "temporal":
            x = x.mean(dim=1)
            return self.fc(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__("ArchitectAgent")
        self.processed_dir = "data/processed"
        self.model_dir = "models/saved"
        self.output_dir = "outputs/reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        self.listen("models_trained")
        self.execute_scratch_implementations()
        self.evaluate_manual_shapes()
        self.train_and_ablate_hybrid()
        self.emit("evaluation_done")

    def execute_scratch_implementations(self):
        def manual_cross_correlation2d(X, K):
            h_x, w_x = X.shape
            h_k, w_k = K.shape
            h_o, w_o = h_x - h_k + 1, w_x - w_k + 1
            Y = np.zeros((h_o, w_o))
            for i in range(h_o):
                for j in range(w_o):
                    Y[i, j] = np.sum(X[i:i+h_k, j:j+w_k] * K)
            return Y

        def manual_pooling(X, pool_size=2, mode="max"):
            h_x, w_x = X.shape
            h_o, w_o = h_x // pool_size, w_x // pool_size
            Y = np.zeros((h_o, w_o))
            for i in range(h_o):
                for j in range(w_o):
                    patch = X[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                    Y[i, j] = np.max(patch) if mode == "max" else np.mean(patch)
            return Y

        X_mock = np.random.randn(10, 10).astype(np.float32)
        K_mock = np.random.randn(3, 3).astype(np.float32)
        y_conv_scratch = manual_cross_correlation2d(X_mock, K_mock)
        y_max_scratch = manual_pooling(X_mock, 2, "max")
        y_avg_scratch = manual_pooling(X_mock, 2, "avg")
        t_X = torch.tensor(X_mock).unsqueeze(0).unsqueeze(0)
        t_K = torch.tensor(K_mock).unsqueeze(0).unsqueeze(0)
        conv_py = nn.functional.conv2d(t_X, t_K).squeeze().numpy()
        max_py = nn.functional.max_pool2d(t_X, 2).squeeze().numpy()
        avg_py = nn.functional.avg_pool2d(t_X, 2).squeeze().numpy()
        diffs = {
            "conv_diff": float(np.abs(y_conv_scratch - conv_py).max()),
            "max_diff": float(np.abs(y_max_scratch - max_py).max()),
            "avg_diff": float(np.abs(y_avg_scratch - avg_py).max())
        }
        with open(os.path.join(self.output_dir, "scratch_verification.json"), "w") as f:
            json.dump(diffs, f)

    def evaluate_manual_shapes(self):
        h_in, w_in = 80, 80
        padding, stride, kernel = 0, 1, 5
        h_conv = int((h_in + 2 * padding - kernel) / stride) + 1
        w_conv = int((w_in + 2 * padding - kernel) / stride) + 1
        h_pool, w_pool = h_conv // 2, w_conv // 2
        report = {
            "theoretical_conv": [h_conv, w_conv],
            "theoretical_pool": [h_pool, w_pool]
        }
        with open(os.path.join(self.output_dir, "dimension_checks.json"), "w") as f:
            json.dump(report, f)

    def train_and_ablate_hybrid(self):
        X_train, y_train = torch.load(os.path.join(self.processed_dir, "seq_train.pt"))
        X_test, y_test = torch.load(os.path.join(self.processed_dir, "seq_test.pt"))
        with open(os.path.join(self.processed_dir, "vocabs.json"), "r") as f:
            vocabs = json.load(f)
        vocab_size = len(vocabs["src_vocab"])
        tgt_vocab_size = len(vocabs["tgt_vocab"])
        train_loader = DataLoader(TensorDataset(X_train, y_train[:, 1]), batch_size=16, shuffle=True)
        results = {}
        for mode in [None, "spatial", "temporal"]:
            model = HybridCNNLSTM(vocab_size, 64, 32, 64, tgt_vocab_size, ablation_mode=mode)
            model = self.to_device(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.002)
            for epoch in range(3):
                model.train()
                for bx, by in train_loader:
                    bx, by = self.to_device(bx), self.to_device(by)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                tx, ty = self.to_device(X_test), self.to_device(y_test[:, 1])
                test_loss = criterion(model(tx), ty).item()
                perplexity = np.exp(test_loss)
                results[str(mode)] = {"loss": test_loss, "perplexity": perplexity}
                if mode is None:
                    torch.save(model.state_dict(), os.path.join(self.model_dir, "hybrid_net.pt"))
        with open(os.path.join(self.output_dir, "ablation_study.json"), "w") as f:
            json.dump(results, f)