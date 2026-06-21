import os
import time
import torch

class BaseAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.state_dir = ".system_states"
        os.makedirs(self.state_dir, exist_ok=True)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def listen(self, trigger_signal):
        signal_path = os.path.join(self.state_dir, trigger_signal)
        while not os.path.exists(signal_path):
            time.sleep(1)
        try:
            os.remove(signal_path)
        except FileNotFoundError:
            pass

    def emit(self, next_signal):
        signal_path = os.path.join(self.state_dir, next_signal)
        with open(signal_path, "w") as f:
            f.write(f"Signal emitted by {self.agent_name}")

    def to_device(self, obj):
        if isinstance(obj, (list, tuple)):
            return [self.to_device(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, torch.nn.Module):
            return obj.to(self.device)
        return obj