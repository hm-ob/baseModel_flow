import torch
import torch.nn as nn
import torchaudio
from transformers import BertConfig, BertModel
from configs.base import Config

# Text Encoder
def build_bert_encoder() -> nn.Module:
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert

# Audio Encoder
class HubertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HubertBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features # (batch, frames, dim)
    
def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    return HubertBase()

def build_audio_encoder(cfg: Config) -> nn.Module:
    # A function to build audio encoder
    # Args: cfg (Config): Config object
    # Returns: nn.Module: Audio encoder
    type = cfg.audio_encoder_type

    encoders = {
        "hubert_base": build_hubert_base_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](cfg)

def build_text_encoder(type: str = "bert") -> nn.Module:
    # A function to build text encoder
    # Args: type (str, optional): Type of text encoder. Defaults to "bert"
    # Returns: torch.nn.Module: Text encoder
    encoders = {
        "bert": build_bert_encoder
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()