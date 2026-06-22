import os
import json
import pandas as pd
import numpy as np
import torch
from agents.base_agent import BaseAgent

class DataStewardAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataStewardAgent")
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        os.makedirs(self.processed_dir, exist_ok=True)

    def run(self):
        self.listen("start_pipeline")
        self.process_tabular()
        self.process_images()
        self.process_sequences()
        self.emit("data_ready")

    def process_tabular(self):
        df = pd.read_csv(os.path.join(self.raw_dir, "PCS Inspections Data - Initial and Follow.csv"))
        df = df.dropna(subset=["Detention"])
        drop_cols = ["IMO number", "Ship Name", "Callsign", "MMSI", "Date"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        df["Detention"] = df["Detention"].map({"no": 0, "yes": 1, 0: 0, 1: 1}).fillna(0).astype(int)
        cat_cols = ["Place", "Flag", "Ship Risk Profile"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
        print("Ordre des colonnes pour LIME :")
        feature_names = df.drop(columns=["Detention"]).columns.tolist()
        for i, name in enumerate(feature_names):
            print(f"Feature_{i} : {name}")
        X = df.drop(columns=["Detention"]).values.astype(np.float32)
        y = df["Detention"].values.astype(np.int64)
        n = len(X)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.64 * n)]
        val_idx = indices[int(0.64 * n):int(0.80 * n)]
        test_idx = indices[int(0.80 * n):]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        torch.save((torch.tensor(X_train), torch.tensor(y_train)), os.path.join(self.processed_dir, "tabular_train.pt"))
        torch.save((torch.tensor(X_val), torch.tensor(y_val)), os.path.join(self.processed_dir, "tabular_val.pt"))
        torch.save((torch.tensor(X_test), torch.tensor(y_test)), os.path.join(self.processed_dir, "tabular_test.pt"))
        np.save(os.path.join(self.processed_dir, "tabular_scaler.npy"), {"mean": mean, "std": std})

    def process_images(self):
        with open(os.path.join(self.raw_dir, "shipsnet.json"), "r") as f:
            dataset = json.load(f)
        data = np.array(dataset["data"], dtype=np.float32) / 255.0
        labels = np.array(dataset["labels"], dtype=np.int64)
        images = data.reshape(-1, 3, 80, 80)
        n = len(images)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.64 * n)]
        val_idx = indices[int(0.64 * n):int(0.80 * n)]
        test_idx = indices[int(0.80 * n):]
        torch.save((torch.tensor(images[train_idx]), torch.tensor(labels[train_idx])), os.path.join(self.processed_dir, "images_train.pt"))
        torch.save((torch.tensor(images[val_idx]), torch.tensor(labels[val_idx])), os.path.join(self.processed_dir, "images_val.pt"))
        torch.save((torch.tensor(images[test_idx]), torch.tensor(labels[test_idx])), os.path.join(self.processed_dir, "images_test.pt"))

    def process_sequences(self):
        texts = []
        labels = []
        with open(os.path.join(self.raw_dir, "maib_data.json"), "r") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item["text"].lower().replace(r"\/", "/"))
                labels.append(item["label"].lower())
        src_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        tgt_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for text in texts:
            for token in text.split():
                if token not in src_vocab:
                    src_vocab[token] = len(src_vocab)
        for label in labels:
            for token in label.split():
                if token not in tgt_vocab:
                    tgt_vocab[token] = len(tgt_vocab)
        src_max_len = max(len(t.split()) for t in texts) + 2
        tgt_max_len = max(len(l.split()) for l in labels) + 2
        src_sequences = []
        tgt_sequences = []
        for text in texts:
            seq = [src_vocab["<SOS>"]] + [src_vocab.get(token, src_vocab["<UNK>"]) for token in text.split()] + [src_vocab["<EOS>"]]
            seq += [src_vocab["<PAD>"]] * (src_max_len - len(seq))
            src_sequences.append(seq)
        for label in labels:
            seq = [tgt_vocab["<SOS>"]] + [tgt_vocab.get(token, tgt_vocab["<UNK>"]) for token in label.split()] + [tgt_vocab["<EOS>"]]
            seq += [tgt_vocab["<PAD>"]] * (tgt_max_len - len(seq))
            tgt_sequences.append(seq)
        src_sequences = np.array(src_sequences, dtype=np.int64)
        tgt_sequences = np.array(tgt_sequences, dtype=np.int64)
        n = len(src_sequences)
        indices = np.random.permutation(n)
        train_idx = indices[:int(0.64 * n)]
        val_idx = indices[int(0.64 * n):int(0.80 * n)]
        test_idx = indices[int(0.80 * n):]
        torch.save((torch.tensor(src_sequences[train_idx]), torch.tensor(tgt_sequences[train_idx])), os.path.join(self.processed_dir, "seq_train.pt"))
        torch.save((torch.tensor(src_sequences[val_idx]), torch.tensor(tgt_sequences[val_idx])), os.path.join(self.processed_dir, "seq_val.pt"))
        torch.save((torch.tensor(src_sequences[test_idx]), torch.tensor(tgt_sequences[test_idx])), os.path.join(self.processed_dir, "seq_test.pt"))
        with open(os.path.join(self.processed_dir, "vocabs.json"), "w") as f:
            json.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab, "src_max_len": src_max_len, "tgt_max_len": tgt_max_len}, f)