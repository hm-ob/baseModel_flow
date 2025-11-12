import argparse
import os
import pickle
import logging
import tqdm
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import random
import re

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def preprocess_MSP_Podcast(args):
    csv_path = os.path.join(args.data_root, "Labels", "labels.txt")
    audio_root = os.path.join(args.data_root, "Audios")
    text_root = os.path.join(args.data_root, "Transcripts")

    samples = []
    missing_audio = 0
    missing_text = 0

    # --- labels.txtから集約ラベルを抽出 ---
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm.tqdm(lines):
        line = line.strip()
        if not line or not line.endswith(";"):
            continue

        # 音声ファイル名＋集約VADラベル
        match = re.match(r"(MSP-PODCAST_\d+_\d+\.wav).*?A:(.*?); V:(.*?); D:(.*?);", line)
        if match:
            fname = match.group(1)
            arousal = float(match.group(2))
            valence = float(match.group(3))
            dominance = float(match.group(4))

            # --- 0〜1に正規化 ---
            #vad = [(valence - 1) / 6, (arousal - 1) / 6, (dominance - 1) / 6]

            vad = [valence, arousal, dominance]
        else:
            continue

        wav_path = os.path.join(audio_root, fname)
        txt_path = os.path.join(text_root, fname.replace(".wav", ".txt"))

        if not os.path.exists(wav_path):
            missing_audio += 1
            continue
        if not os.path.exists(txt_path):
            missing_text += 1
            continue

        # 音声長チェック
        wav_data, sr = sf.read(wav_path, dtype="float32")
        if len(wav_data) < args.ignore_length:
            continue

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        samples.append((fname, wav_path, text, vad))

    logging.info(f"Total usable samples: {len(samples)}")
    logging.info(f"Missing audios: {missing_audio}, Missing transcripts: {missing_text}")

    # --- データ分割 ---
    partition_path = os.path.join(args.data_root, "Partitions.txt")
    split_map = {}
    with open(partition_path, "r") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != 2:
                continue
            split_name = parts[0].strip().lower()
            if split_name == "development":
                split_name = "val"
            filename = parts[1].strip()
            split_map[filename] = split_name

    samples_dict = {"train": [], "val": [], "test1": [], "test2": [], "test3": []}

    for fname, wav_path, text, vad in samples:
        split_name = split_map.get(fname)
        if split_name is None:
            continue
        if split_name not in samples_dict:
            samples_dict[split_name] = []
        samples_dict[split_name].append((wav_path, text, vad))

    save_root = args.dataset + "_preprocessed"
    os.makedirs(save_root, exist_ok=True)
    for split_name, samples in samples_dict.items():
        out_path = os.path.join(save_root, f"{split_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(samples, f)
        logging.info(f"{split_name}: {len(samples)} samples saved to {out_path}")

    logging.info("Preprocessing completed successfully.")


def main(args):
    preprocess_MSP_Podcast(args)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds", "--dataset", type=str, default="MSP-Podcast", help="Dataset name"
    )
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        required=True,
        help="Path to MSP-Podcast dataset root directory",
    )
    parser.add_argument(
        "-ignore_length",
        type=int,
        default=16000,
        help="Ignore audio samples shorter than this length (in samples)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
