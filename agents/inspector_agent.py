import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from agents.base_agent import BaseAgent
from agents.scientist_agent import MLPCustom, LeNet5Variant, Seq2SeqEncoder, Seq2SeqDecoder
from bert_score import score

class InspectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("InspectorAgent")
        self.processed_dir = "data/processed"
        self.model_dir = "models/saved"
        self.output_dir = "outputs/reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        self.listen("evaluation_done")
        with open(os.path.join(self.processed_dir, "vocabs.json"), "r") as f:
            vocabs = json.load(f)
        self.src_vocab = vocabs["src_vocab"]
        self.tgt_vocab = vocabs["tgt_vocab"]
        self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}
        self.mlp = MLPCustom(4, 64, 2)
        self.mlp.load_state_dict(torch.load(os.path.join(self.model_dir, "best_mlp.pt")))
        self.mlp = self.to_device(self.mlp)
        self.mlp.eval()
        self.cnn = LeNet5Variant()
        self.cnn.load_state_dict(torch.load(os.path.join(self.model_dir, "best_cnn.pt")))
        self.cnn = self.to_device(self.cnn)
        self.cnn.eval()
        self.encoder = Seq2SeqEncoder(len(self.src_vocab), 64, 128)
        self.decoder = Seq2SeqDecoder(len(self.tgt_vocab), 64, 128)
        self.encoder.load_state_dict(torch.load(os.path.join(self.model_dir, "seq2seq_encoder.pt")))
        self.decoder.load_state_dict(torch.load(os.path.join(self.model_dir, "seq2seq_decoder.pt")))
        self.encoder, self.decoder = self.to_device(self.encoder), self.to_device(self.decoder)
        self.encoder.eval()
        self.decoder.eval()
        self.scaler = np.load(os.path.join(self.processed_dir, "tabular_scaler.npy"), allow_pickle=True).item()
        self.evaluate_metrics_and_save()
        self.generate_feature_maps()
        self.generate_xai_explanations()
        self.serve_demo_simulation()
        self.emit("pipeline_complete")

    def evaluate_metrics_and_save(self):
        X_test, y_test = torch.load(os.path.join(self.processed_dir, "tabular_test.pt"))
        with torch.no_grad():
            probabilities = torch.softmax(self.mlp(self.to_device(X_test)), dim=1).cpu().numpy()
            preds = (probabilities[:, 1] >= 0.35).astype(int)
        y_true = y_test.numpy()
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        report = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]]
        }
        with open(os.path.join(self.output_dir, "tabular_performance.json"), "w") as f:
            json.dump(report, f)
        X_img, y_img = torch.load(os.path.join(self.processed_dir, "images_test.pt"))
        with torch.no_grad():
            img_preds = torch.argmax(self.cnn(self.to_device(X_img)), dim=1).cpu().numpy()
        img_true = y_img.numpy()
        img_acc = float(np.mean(img_preds == img_true))
        with open(os.path.join(self.output_dir, "vision_performance.json"), "w") as f:
            json.dump({"accuracy": img_acc}, f)

    def generate_feature_maps(self):
        X_img, _ = torch.load(os.path.join(self.processed_dir, "images_test.pt"))
        sample = self.to_device(X_img[0:1])
        with torch.no_grad():
            f_map = torch.relu(self.cnn.conv1(sample)).cpu().numpy()
        fig, axes = plt.subplots(1, f_map.shape[1], figsize=(12, 3))
        for i in range(f_map.shape[1]):
            axes[i].imshow(f_map[0, i], cmap="viridis")
            axes[i].axis("off")
        plt.savefig(os.path.join(self.output_dir, "feature_maps.png"))
        plt.close()

    def generate_xai_explanations(self):
        X_train, _ = torch.load(os.path.join(self.processed_dir, "tabular_train.pt"))
        X_test, _ = torch.load(os.path.join(self.processed_dir, "tabular_test.pt"))
        def model_predict_wrapper(x_np):
            t = self.to_device(torch.tensor(x_np, dtype=torch.float32))
            with torch.no_grad():
                return torch.softmax(self.mlp(t), dim=1).cpu().numpy()
        train_sample = X_train.numpy()[:50]
        test_sample = X_test.numpy()[:5]
        explainer_shap = shap.KernelExplainer(model_predict_wrapper, train_sample)
        shap_obj = explainer_shap.shap_values(test_sample)
        if isinstance(shap_obj, list):
            chosen_shap_values = shap_obj[1]
        elif isinstance(shap_obj, np.ndarray) and len(shap_obj.shape) == 3:
            chosen_shap_values = shap_obj[:, :, 1]
        else:
            chosen_shap_values = shap_obj
        plt.figure()
        shap.summary_plot(chosen_shap_values, test_sample, show=False)
        plt.savefig(os.path.join(self.output_dir, "shap_summary.png"))
        plt.close()
        explainer_lime = lime_tabular.LimeTabularExplainer(
            train_sample, 
            mode="classification", 
            feature_names=["Feature_0", "Feature_1", "Feature_2", "Feature_3"]
        )
        exp = explainer_lime.explain_instance(test_sample[0], model_predict_wrapper, num_features=4)
        exp.save_to_file(os.path.join(self.output_dir, "lime_explanation.html"))

    def compute_bleu_score(self, reference, candidate):
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        if len(cand_tokens) == 0:
            return 0.0
        matches = sum(1 for t in cand_tokens if t in ref_tokens)
        return matches / len(cand_tokens)
    
    def compute_semantic_score(self, candidates, references):    
        P, R, F1 = score(candidates, references, lang="en", verbose=True)
        return F1.mean().item()

    def serve_demo_simulation(self):
        mock_image = torch.randn(1, 3, 80, 80)
        mock_tabular = np.array([[12, 45, 2, 0]], dtype=np.float32)
        mock_text = "port tender vessel suffered an engine failure"
        mock_image = self.to_device(mock_image)
        with torch.no_grad():
            vision_out = torch.softmax(self.cnn(mock_image), dim=1)
            ship_detected = torch.argmax(vision_out, dim=1).item()
        scaled_tabular = (mock_tabular - self.scaler["mean"]) / self.scaler["std"]
        tab_tensor = self.to_device(torch.tensor(scaled_tabular, dtype=torch.float32))
        with torch.no_grad():
            mlp_out = torch.softmax(self.mlp(tab_tensor), dim=1)
            raw_prob = mlp_out[0, 1].item()
            detention_prob = raw_prob if raw_prob > 0.35 else raw_prob * 0.1
        tokens = [self.src_vocab["<SOS>"]] + [self.src_vocab.get(w, self.src_vocab["<UNK>"]) for w in mock_text.split()] + [self.src_vocab["<EOS>"]]
        text_tensor = self.to_device(torch.tensor([tokens], dtype=torch.int64))
        generated_diagnostic = []
        with torch.no_grad():
            hidden = self.encoder(text_tensor)
            dec_input = self.to_device(torch.tensor([self.tgt_vocab["<SOS>"]], dtype=torch.int64))
            for _ in range(10):
                output, hidden = self.decoder(dec_input, hidden)
                pred = torch.argmax(output, dim=1).item()
                if pred == self.tgt_vocab["<EOS>"]:
                    break
                generated_diagnostic.append(self.inv_tgt_vocab.get(pred, "<UNK>"))
                dec_input = self.to_device(torch.tensor([pred], dtype=torch.int64))
        candidate_sentence = " ".join(generated_diagnostic)
        bleu = self.compute_bleu_score(mock_text, candidate_sentence)
        bert_score = self.compute_semantic_score([candidate_sentence], [mock_text])
        demo_results = {
            "ship_detection_node": ship_detected,
            "detention_probability": detention_prob,
            "generated_text": candidate_sentence,
            "simulated_bleu_score": bleu,
            "simulated_bert_score": bert_score
        }
        with open(os.path.join(self.output_dir, "demo_simulation_output.json"), "w") as f:
            json.dump(demo_results, f)