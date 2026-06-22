import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from agents.base_agent import BaseAgent
from sklearn.preprocessing import StandardScaler

class MLPSequential(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.network(x)

class MLPCustom(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

class LeNet5Variant(nn.Module):
    def __init__(self, num_classes=2, padding=0, stride=1, use_avg_pooling=False, conv1_filters=6, conv2_filters=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_filters, kernel_size=5, stride=stride, padding=padding)
        self.pool = nn.AvgPool2d(2, 2) if use_avg_pooling else nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(conv2_filters, conv2_filters, kernel_size=1)
        self.flattened_dim = self._get_conv_output_dim(conv1_filters, conv2_filters)
        self.fc1 = nn.Linear(self.flattened_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _get_conv_output_dim(self, conv1_filters, conv2_filters):
        with torch.no_grad():
            x = torch.zeros(1, 3, 80, 80)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv1x1(self.conv2(x))))
            return x.numel()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv1x1(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RNNAngularLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMAngularLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUAngularLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        out, hidden = self.gru(embedded, hidden)
        out = self.fc(out.squeeze(1))
        return out, hidden

class ScientistAgent(BaseAgent):
    def __init__(self):
        super().__init__("ScientistAgent")
        self.processed_dir = "data/processed"
        self.model_dir = "models/saved"
        self.output_dir = "outputs/reports"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        self.listen("data_ready")
        self.train_tabular_models()
        self.train_image_models()
        self.train_sequential_models()
        self.emit("models_trained")

    def init_weights(self, model, strategy):
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if strategy == "gaussian":
                    nn.init.normal_(param, mean=0.0, std=0.01)
                elif strategy == "constant":
                    nn.init.constant_(param, 0.1)
                elif strategy == "xavier":
                    nn.init.xavier_normal_(param)

    def plot_learning_curves(self, train_losses, val_losses, model_name):
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label="Entrainement", color='blue', marker='o')
            
            if len(val_losses) > 0:
                plt.plot(val_losses, label="Validation", color='red', marker='x')
            
            plt.xlabel("Epoques")
            plt.ylabel("Perte")
            plt.title(f"Courbe d'apprentissage : {model_name}")
            plt.legend()
            plt.grid(True)
            
    
            save_path = os.path.join(self.output_dir, f"{model_name}_learning_curve.png")
            plt.savefig(save_path)
            plt.close()
            

    def train_tabular_models(self):
        X_train, y_train = torch.load(os.path.join(self.processed_dir, "tabular_train.pt"))
        X_val, y_val = torch.load(os.path.join(self.processed_dir, "tabular_val.pt"))
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train.numpy()), dtype=torch.float32)
        X_val = torch.tensor(scaler.transform(X_val.numpy()), dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        weights = torch.tensor([1.0, 5.0], device=self.device)
        best_val_loss = float("inf")
        best_state = None
        tabular_reports = {}

        for init_strategy in ["gaussian", "constant", "xavier"]:
            model = MLPCustom(X_train.shape[1], 64, 2)
            self.init_weights(model, init_strategy)
            model = self.to_device(model)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_losses = []
            val_losses = []
            for epoch in range(5):
                model.train()
                epoch_train_loss = 0.0
                steps = 0
                for bx, by in train_loader:
                    bx, by = self.to_device(bx), self.to_device(by)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    steps += 1
                train_losses.append(epoch_train_loss / max(1, steps))
                model.eval()
                with torch.no_grad():
                    val_x, val_y = self.to_device(X_val), self.to_device(y_val)
                    v_loss = criterion(model(val_x), val_y).item()
                    val_losses.append(v_loss)
            self.plot_learning_curves(train_losses, val_losses, f"MLP_{init_strategy}")
            tabular_reports[init_strategy] = val_losses[-1]
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_state = model.state_dict()
        torch.save(best_state, os.path.join(self.model_dir, "best_mlp.pt"))
        seq_model = MLPSequential(X_train.shape[1], 64, 2)
        seq_model = self.to_device(seq_model)
        optimizer_seq = optim.Adam(seq_model.parameters(), lr=0.001)
        criterion_seq = nn.CrossEntropyLoss(weight=weights)
        for epoch in range(5):
            seq_model.train()
            for bx, by in train_loader:
                bx, by = self.to_device(bx), self.to_device(by)
                optimizer_seq.zero_grad()
                loss = criterion_seq(seq_model(bx), by)
                loss.backward()
                optimizer_seq.step()
        torch.save(seq_model.state_dict(), os.path.join(self.model_dir, "sequential_mlp.pt"))
        with open(os.path.join(self.output_dir, "tabular_initializations.json"), "w") as f:
            json.dump(tabular_reports, f)

    def train_image_models(self):
        X_train, y_train = torch.load(os.path.join(self.processed_dir, "images_train.pt"))
        X_val, y_val = torch.load(os.path.join(self.processed_dir, "images_val.pt"))
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
        cnn_comparison = {}
        for config in [{"padding": 0, "stride": 1, "pool": False}, {"padding": 1, "stride": 2, "pool": True}]:
            model = LeNet5Variant(padding=config["padding"], stride=config["stride"], use_avg_pooling=config["pool"])
            model = self.to_device(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_losses = []
            val_losses = []
            for epoch in range(2):
                model.train()
                epoch_train_loss = 0.0
                steps = 0
                for bx, by in train_loader:
                    bx, by = self.to_device(bx), self.to_device(by)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    steps += 1
                train_losses.append(epoch_train_loss / max(1, steps))
                model.eval()
                epoch_val_loss = 0.0
                v_steps = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = self.to_device(vx), self.to_device(vy)
                        v_loss = criterion(model(vx), vy)
                        epoch_val_loss += v_loss.item()
                        v_steps += 1
                val_losses.append(epoch_val_loss / max(1, v_steps))
            config_name = f"pad_{config['padding']}_stride_{config['stride']}_avg_{config['pool']}"
            self.plot_learning_curves(train_losses, val_losses, f"CNN_{config_name}")
            cnn_comparison[config_name] = val_losses[-1]
            if config["padding"] == 0 and config["stride"] == 1 and not config["pool"]:
                torch.save(model.state_dict(), os.path.join(self.model_dir, "best_cnn.pt"))
        with open(os.path.join(self.output_dir, "cnn_hyperparameters.json"), "w") as f:
            json.dump(cnn_comparison, f)

        filter_configurations = [
            {"conv1": 6, "conv2": 16, "name": "Configuration_Standard"},
            {"conv1": 12, "conv2": 32, "name": "Configuration_Large"}
        ]
        filter_experiment_results = {}
        for config in filter_configurations:
            model = LeNet5Variant(num_classes=2, conv1_filters=config["conv1"], conv2_filters=config["conv2"])
            model = self.to_device(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(2):
                model.train()
                for bx, by in train_loader:
                    bx, by = self.to_device(bx), self.to_device(by)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = self.to_device(vx), self.to_device(vy)
                    outputs = model(vx)
                    _, predicted = torch.max(outputs.data, 1)
                    total += vy.size(0)
                    correct += (predicted == vy).sum().item()
            accuracy = correct / total if total > 0 else 0
            filter_experiment_results[config["name"]] = {
                "conv1_filters": config["conv1"],
                "conv2_filters": config["conv2"],
                "accuracy": accuracy
            }
        with open(os.path.join(self.output_dir, "cnn_filter_experiment.json"), "w") as f:
            json.dump(filter_experiment_results, f)

    def train_sequential_models(self):
        X_train, y_train = torch.load(os.path.join(self.processed_dir, "seq_train.pt"))
        X_test, _ = torch.load(os.path.join(self.processed_dir, "seq_test.pt"))
        with open(os.path.join(self.processed_dir, "vocabs.json"), "r") as f:
            vocabs = json.load(f)
        src_vocab_size = len(vocabs["src_vocab"])
        tgt_vocab_size = len(vocabs["tgt_vocab"])
        train_loader_seq = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        test_loader_seq = DataLoader(TensorDataset(X_test, torch.zeros_like(X_test)), batch_size=16, shuffle=False)
        sequence_comparison_results = {}
        for cell_type in ["rnn", "lstm", "gru"]:
            if cell_type == "rnn":
                lm_model = RNNAngularLM(src_vocab_size, 64, 128)
            elif cell_type == "lstm":
                lm_model = LSTMAngularLM(src_vocab_size, 64, 128)
            elif cell_type == "gru":
                lm_model = GRUAngularLM(src_vocab_size, 64, 128)
            lm_model = self.to_device(lm_model)
            optimizer = optim.Adam(lm_model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            train_losses = []
            for epoch in range(3):
                lm_model.train()
                epoch_train_loss = 0.0
                steps = 0
                for bx, _ in train_loader_seq:
                    bx = self.to_device(bx)
                    labels = bx[:, 1]
                    inputs = bx[:, :-1]
                    optimizer.zero_grad()
                    outputs = lm_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lm_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    steps += 1
                train_losses.append(epoch_train_loss / max(1, steps))
            self.plot_learning_curves(train_losses, [], f"Seq_{cell_type}")
            lm_model.eval()
            total_test_loss = 0
            steps = 0
            with torch.no_grad():
                for bx, _ in test_loader_seq:
                    bx = self.to_device(bx)
                    labels = bx[:, 1]
                    inputs = bx[:, :-1]
                    outputs = lm_model(inputs)
                    total_test_loss += criterion(outputs, labels).item()
                    steps += 1
            avg_loss = total_test_loss / max(1, steps)
            perplexity = np.exp(avg_loss)
            sequence_comparison_results[cell_type] = {
                "loss": avg_loss,
                "perplexity": float(perplexity)
            }
            torch.save(lm_model.state_dict(), os.path.join(self.model_dir, f"lm_{cell_type}.pt"))
        with open(os.path.join(self.output_dir, "rnn_lstm_gru_comparison.json"), "w") as f:
            json.dump(sequence_comparison_results, f)

        clipping_results = {}
        for clip in [True, False]:
            rnn_model = RNNAngularLM(src_vocab_size, 64, 128)
            rnn_model = self.to_device(rnn_model)
            optimizer = optim.Adam(rnn_model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(3):
                rnn_model.train()
                for bx, _ in train_loader_seq:
                    bx = self.to_device(bx)
                    labels = bx[:, 1]
                    inputs = bx[:, :-1]
                    optimizer.zero_grad()
                    outputs = rnn_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=1.0)
                    optimizer.step()
            rnn_model.eval()
            total_test_loss = 0
            steps = 0
            with torch.no_grad():
                for bx, _ in test_loader_seq:
                    bx = self.to_device(bx)
                    labels = bx[:, 1]
                    inputs = bx[:, :-1]
                    outputs = rnn_model(inputs)
                    total_test_loss += criterion(outputs, labels).item()
                    steps += 1
            avg_loss = total_test_loss / max(1, steps)
            clipping_results[f"with_clipping_{clip}"] = avg_loss
        with open(os.path.join(self.output_dir, "gradient_clipping_experiment.json"), "w") as f:
            json.dump(clipping_results, f)

        encoder = Seq2SeqEncoder(src_vocab_size, 64, 128)
        decoder = Seq2SeqDecoder(tgt_vocab_size, 64, 128)
        encoder, decoder = self.to_device(encoder), self.to_device(decoder)
        enc_opt = optim.Adam(encoder.parameters(), lr=0.001)
        dec_opt = optim.Adam(decoder.parameters(), lr=0.001)
        criterion_seq = nn.CrossEntropyLoss(ignore_index=0)
        train_losses = []
        for epoch in range(3):
            encoder.train()
            decoder.train()
            epoch_loss = 0.0
            steps = 0
            for src_b, tgt_b in train_loader_seq:
                src_b, tgt_b = self.to_device(src_b), self.to_device(tgt_b)
                enc_opt.zero_grad()
                dec_opt.zero_grad()
                
                hidden = encoder(src_b)
                iter_loss = 0.0 
                dec_input = tgt_b[:, 0]
                
                for t in range(1, tgt_b.size(1)):
                    output, hidden = decoder(dec_input, hidden)
                    l = criterion_seq(output, tgt_b[:, t])
                    l.backward(retain_graph=True)
                    iter_loss += l.item()
                    
                    if np.random.rand() < 0.5:
                        dec_input = torch.argmax(output, dim=1) 
                    else:
                        dec_input = tgt_b[:, t] 
                
                enc_opt.step()
                dec_opt.step()
                epoch_loss += (iter_loss / max(1, tgt_b.size(1) - 1))
                steps += 1
            train_losses.append(epoch_loss / max(1, steps))
        
        torch.save(encoder.state_dict(), os.path.join(self.model_dir, "seq2seq_encoder.pt"))
        torch.save(decoder.state_dict(), os.path.join(self.model_dir, "seq2seq_decoder.pt"))