import torch
import torch.nn as nn
from configs.base import Config
from .modules import build_audio_encoder, build_text_encoder
try:
    from src.tflow_utils import TransformerGlow
except Exception:
    TransformerGlow = None

class MemoCMT(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(MemoCMT, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
        # If using TransformerGlow, freeze its transformer and control glow params
        if TransformerGlow is not None and isinstance(self.text_encoder, TransformerGlow):
            # freeze transformer params
            for param in self.text_encoder.transformer.parameters():
                param.requires_grad = False
            # control glow training via cfg.text_unfreeze
            for param in self.text_encoder.glow.parameters():
                param.requires_grad = cfg.text_unfreeze
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = cfg.text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        # Fusion module
        self.text_attention = nn.MultiheadAttention(
            embed_dim = cfg.text_encoder_dim,
            num_heads = cfg.num_attention_head,
            dropout = cfg.dropout,
            batch_first = True,
        )
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)
        self.dropout = nn.Dropout(cfg.dropout) # 双方向なら消す

        self.audio_attention = nn.MultiheadAttention(
            embed_dim = cfg.audio_encoder_dim,
            num_heads = cfg.num_attention_head,
            dropout = cfg.dropout,
            batch_first = True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        """self.fusion_attention = nn.MultiheadAttention(
            embed_dim = cfg.fusion_dim,
            num_heads = cfg.num_attention_head,
            dropout = cfg.dropout,
            batch_first = True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)"""

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.fusion_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer
        
        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):
        # input_text can be either:
        # - tensor of input_ids (old behaviour, no attention mask)
        # - tuple (input_ids, attention_mask)
        if isinstance(input_text, (tuple, list)):
            input_ids, attention_mask = input_text
        else:
            input_ids = input_text
            attention_mask = None

        # For BERT model, call and compute mean over tokens
        if TransformerGlow is not None and isinstance(self.text_encoder, TransformerGlow):
            # TransformerGlow expects input_ids and attention_mask
            if attention_mask is None:
                # create attention mask (assume non-zero tokens are real)
                attention_mask = (input_ids != 0).long()
            # text_encoder returns z (bsz, dim) or (z, loss)
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
            if isinstance(out, tuple):
                text_z = out[0]
            else:
                text_z = out
            text_embeddings = text_z.unsqueeze(1)  # (bsz, 1, dim)
        else:
            hidden = self.text_encoder(input_ids)
            # hidden can be a ModelOutput with last_hidden_state
            if hasattr(hidden, "last_hidden_state"):
                text_embeddings = hidden.last_hidden_state.mean(dim=1).unsqueeze(1)
            else:
                # fallback: assume tensor
                text_embeddings = hidden.mean(dim=1).unsqueeze(1)
        if len(input_audio.size()) != 2: # データが長い場合
            batch_size, num_samples = input_audio.seze(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            # audio_embeddings = audio_embeddings.mean(1)
            audio_embeddings = audio_embeddings.view(
                batch_size, num_samples, *audio_embeddings.shape[1:] 
            )
        else:
            audio_embeddings = self.audio_encoder(input_audio)

        ## Fusion Module
        # Text cross attention text Q audio, K and V text
        """text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            average_attn_weights = False,
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)
        text_norm = self.dropout(text_norm)"""

        # Audio cross attention Q text, K and V audio
        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            average_attn_weights = False,
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)
        audio_norm = self.dropout(audio_norm)

        # Concatenate the text and audio embeddings
        """fusion_norm = torch.cat((text_norm, audio_norm), 1)
        fusion_norm = self.dropout(fusion_norm)"""

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = audio_norm[:, 0, :] # 双方向ならfusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = audio_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = audio_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = audio_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.gelu(x)
        x = self.dropout(x)
        out = self.classifer(x)
        out = torch.sigmoid(out)

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [
                #text_attn_output_weights,
                audio_attn_output_weights,
            ]

        return out, cls_token_final_fusion_norm, audio_norm #, text_norm

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        # Keep compatibility: accept either input_ids tensor or tuple (input_ids, attention_mask)
        if isinstance(input_ids, (tuple, list)):
            input_ids, attention_mask = input_ids
        else:
            attention_mask = None

        if TransformerGlow is not None and isinstance(self.text_encoder, TransformerGlow):
            if attention_mask is None:
                attention_mask = (input_ids != 0).long()
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_loss=False)
            if isinstance(out, tuple):
                return out[0]
            return out
        else:
            hidden = self.text_encoder(input_ids)
            if hasattr(hidden, "last_hidden_state"):
                return hidden.last_hidden_state
            return hidden