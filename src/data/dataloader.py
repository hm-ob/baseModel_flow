import os
import pickle
from typing import Tuple, Union
import re
import logging
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torchaudio

from models.networks import MemoCMT
from configs.base import Config
#from torchvggish.vgish_input import waveform_to_examples
from tqdm.auto import tqdm
import pickle
import librosa

class BaseDataset(Dataset):
    def __init__(
            self,
            cfg: Config,
            data_mode: str = "train.pkl",
            encoder_model: Union[MemoCMT, None] = None,
    ):
        super(BaseDataset, self).__init__()

        with open(os.path.join(cfg.data_root, data_mode), "rb") as train_file:
            self.data_list = pickle.load(train_file)

        if cfg.text_encoder_type in ("bert", "bert-flow"):
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise NotImplementedError("Tokenizer {} is not implemented".format(cfg.text_encoder_type))
        
        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        if cfg.batch_size == 1:
            self.audio_max_length = None
            self.text_max_length = None
        
        self.audio_encoder_type = cfg.audio_encoder_type

        self.encode_data = False
        self.list_encode_audio_data = []
        self.list_encode_text_data = []
        if encoder_model is not None:
            self._encode_data(encoder_model)
            self.encode_data = True

    def _encode_data(self, encoder):
        logging.info("Encoding data for training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # encoder.train()
        encoder.eval()
        encoder.to(device)
        with torch.no_grad():
            for index in tqdm(range(len(self.data_list))):
                audio_path, text, _ = self.data_list[index]

                samples = self.__paudio__(audio_path)
                audio_embedding = (
                    encoder.encode_audio(samples.unsqueeze(0).to(device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                )
                self.list_encode_audio_data.append(audio_embedding)

                # __ptext__ now returns (input_ids, attention_mask)
                input_ids, attention_mask = self.__ptext__(text)
                # prepare batch dim
                input_ids_b = input_ids.unsqueeze(0).to(device)
                attention_mask_b = attention_mask.unsqueeze(0).to(device)
                # encoder.encode_text should accept (input_ids, attention_mask) or input_ids only
                try:
                    text_embedding = (
                        encoder.encode_text(input_ids_b, attention_mask_b)
                        .squeeze(0)
                        .detach()
                        .cpu()
                    )
                except TypeError:
                    # fallback: encoder.encode_text(input_ids)
                    text_embedding = (
                        encoder.encode_text(input_ids_b)
                        .squeeze(0)
                        .detach()
                        .cpu()
                    )
                self.list_encode_text_data.append(text_embedding)

    def __getitem__(
            self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        audio_path, text, label = self.data_list[index]
        input_audio = (
            self.list_encode_audio_data[index]
            if self.encode_data
            else self.__paudio__(audio_path)
        )
        input_text = (
            self.list_encode_text_data[index]
            if self.encode_data
            else self.__ptext__(text)
        )
        label = self.__plabel__(label)

        return input_text, input_audio, label
    
    def __paudio__(self, file_path: str) -> torch.Tensor:
        samples, sr = sf.read(file_path, dtype="float32")
        samples = torch.from_numpy(samples)
        samples = torchaudio.functional.resample(samples, sr, 16000)
        if (
            self.audio_max_length is not None
            and samples.shape[0] < self.audio_max_length
        ):
            samples = torch.nn.functional.pad(
                samples, (0, self.audio_max_length - samples.shape[0])
            )
        elif self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]

        return samples
    
    def _text_preprocessing(self, text):
        # Remove '@name'
        text = re.sub("[\(\[].*?[\)\]]", "", text)

        text = re.sub(" +", " ", text).strip()

        # Normalize and clean up text; order matters
        try:
            text = " ".join(text.split()) # clean up whitespaces
        except:
            text = "NULL"

        # Convert empty string to NULL
        if not text.strip():
            text = "NULL"

        return text

    def __ptext__(self, text: str) -> torch.Tensor:
        text = self._text_preprocessing(text)
        # Return both input_ids and attention_mask so models that require masks (e.g., TransformerGlow)
        tokenized = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.text_max_length if self.text_max_length is not None else None,
            padding="max_length" if self.text_max_length is not None else False,
            truncation=True if self.text_max_length is not None else False,
            return_attention_mask=True,
        )
        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(tokenized["attention_mask"], dtype=np.int64)
        return torch.from_numpy(input_ids), torch.from_numpy(attention_mask)
    
    def __plabel__(self, label: Tuple) -> torch.Tensor:
        return torch.tensor(label)
    
    def __len__(self):
        return len(self.data_list)
    

def build_train_test_dataset(cfg: Config, encoder_model: Union[MemoCMT, None] = None):
    DATASET_MAP = {
        "MSP-Podcast": BaseDataset
    }

    dataset = DATASET_MAP.get(cfg.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                cfg.data_name, DATASET_MAP.keys()
            )
        )
    if cfg.data_name in ["MSP-Podcast_MSER"]:
        return dataset(cfg)
    
    train_data = dataset(
        cfg,
        data_mode = "train.pkl",
        encoder_model = encoder_model,
    )

    if encoder_model is not None:
        encoder_model.eval()
    test_set = cfg.data_valid if cfg.data_valid is not None else "test1.pkl"
    test_data = dataset(
        cfg,
        data_mode=test_set,
        encoder_model=encoder_model,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return (train_dataloader, test_dataloader)