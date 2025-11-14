import logging
import os
from typing import Dict

import torch
from torch import Tensor
from configs.base import Config
from models.networks import MemoCMT
from utils.torch.trainer import TorchTrainer

def concordance_correlation_coefficient(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds_mean = preds.mean(dim=0)
    labels_mean = labels.mean(dim=0)
    cov = ((preds - preds_mean) * (labels - labels_mean)).mean(dim=0)
    preds_var = preds.var(dim=0, unbiased=False)
    labels_var = labels.var(dim=0, unbiased=False)
    ccc = (2 * cov) / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
    return ccc


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: MemoCMT,
        criterion: torch.nn.MSELoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        # ラベルの正規化
        label = (label - 1) / 6
        label = label.to(self.device)
        # input_text can be a tensor or a tuple (input_ids, attention_mask)
        if isinstance(input_text, (list, tuple)):
            input_text = tuple(x.to(self.device) for x in input_text)
        else:
            input_text = input_text.to(self.device)

        # Forward pass
        output = self.network(input_text, input_audio)
        output = output[0] # output: (batch_size, num_classes)  change
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        mae = torch.mean(torch.abs(output - label), dim=0)
        mse = torch.mean((output - label) ** 2, dim=0)
        #ccc = concordance_correlation_coefficient(output, label*6 + 1)
        #valence_ccc, arousal_ccc, dominance_ccc = ccc.tolist()
        return {
            "loss": loss.detach().cpu().item(),
            "v_mae": mae[0].detach().cpu().item(),
            "a_mae": mae[1].detach().cpu().item(),
            "d_mae": mae[2].detach().cpu().item(),
            "v_mse": mse[0].detach().cpu().item(),
            "a_mse": mse[1].detach().cpu().item(),
            "d_mse": mse[2].detach().cpu().item(),
            #"v_ccc": valence_ccc,
            #"a_ccc": arousal_ccc,
            #"d_ccc": dominance_ccc,
            #"mean_ccc": float(sum(ccc.tolist()) / 3),
        }

    def test_epoch_start(self):
        self.all_outputs = []
        self.all_labels = []

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = (label-1) / 6
        label = label.to(self.device)
        if isinstance(input_text, (list, tuple)):
            input_text = tuple(x.to(self.device) for x in input_text)
        else:
            input_text = input_text.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            output = output[0]  # output: (batch_size, num_classes)  change
            self.all_outputs.append(output)
            self.all_labels.append(label)
            loss = self.criterion(output, label)
            # Calculate accuracy
            mae = torch.mean(torch.abs(output - label), dim=0)
            mse = torch.mean((output - label) ** 2, dim=0)
            ccc = concordance_correlation_coefficient(output*6 + 1, label*6 + 1)
            ccc = ccc.detach().cpu()
            valence_ccc, arousal_ccc, dominance_ccc = ccc.tolist()
        return {
            "loss": loss.detach().cpu().item(),
            "v_mae": mae[0].detach().cpu().item(),
            "a_mae": mae[1].detach().cpu().item(),
            "d_mae": mae[2].detach().cpu().item(),
            "v_mse": mse[0].detach().cpu().item(),
            "a_mse": mse[1].detach().cpu().item(),
            "d_mse": mse[2].detach().cpu().item(),
            "v_ccc": valence_ccc,
            "a_ccc": arousal_ccc,
            "d_ccc": dominance_ccc,
            "mean_ccc": float(sum(ccc.tolist()) / 3),
        }

    def test_epoch_end(self):
        outputs = torch.cat(self.all_outputs, dim=0)
        labels = torch.cat(self.all_labels, dim=0)

        outputs_orig = outputs * 6 + 1 # Rescale to original VAD range
        labels_orig = labels * 6 + 1
        ccc = concordance_correlation_coefficient(outputs_orig, labels_orig)
        valence_ccc, arousal_ccc, dominance_ccc = ccc.tolist()
        return {
            "v_ccc": valence_ccc,
            "a_ccc": arousal_ccc,
            "d_ccc": dominance_ccc,
            "mean_ccc": float(sum(ccc.tolist()) / 3),
        }