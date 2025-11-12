import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
import csv
import glob
import argparse
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config
from collections import Counter
from typing import Tuple


def concordance_correlation_coefficient(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds_mean = preds.mean(dim=0)
    labels_mean = labels.mean(dim=0)
    cov = ((preds - preds_mean) * (labels - labels_mean)).mean(dim=0)
    preds_var = preds.var(dim=0, unbiased=False)
    labels_var = labels.var(dim=0, unbiased=False)
    ccc = (2 * cov) / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
    return ccc

def eval(cfg, checkpoint_path, output_csv, plot, all_state_dict=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = getattr(networks, cfg.model_type)(cfg)
    network.to(device)

    _, test_ds = build_train_test_dataset(cfg)

    weight = torch.load(checkpoint_path, map_location=device)
    if all_state_dict:
        weight = weight["state_dict_network"]
    network.load_state_dict(weight)
    network.eval()
    network.to(device)

    y_true_list = []
    y_pred_list = []

    for input_ids, audio, label in tqdm(test_ds, desc="Evaluating"):
        input_ids, audio, label = input_ids.to(device), audio.to(device), label.to(device)
        with torch.no_grad():
            output = network(input_ids, audio)[0]
        y_true_list.append(label)
        y_pred_list.append(output)

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)

    # ラベル正規化・逆正規化を行って計算
    mse = torch.mean(((y_true-1)/6 - y_pred) ** 2, dim=0)
    mae = torch.mean(torch.abs((y_true-1)/6 - y_pred), dim=0)
    ccc = concordance_correlation_coefficient((y_pred * 6 + 1), y_true)
    valence_ccc, arousal_ccc, dominance_ccc = ccc.tolist()

    logging.info(f"valence_CCC: {valence_ccc:.4f}")
    logging.info(f"arousal_CCC: {arousal_ccc:.4f}")
    logging.info(f"dominance_CCC: {dominance_ccc:.4f}")
    logging.info("MSE(V,A,D): {}".format(mse.cpu().numpy()))
    logging.info("MAE(V, A, D): {}".format(mae.cpu().numpy()))

    # Save to CSV
    if output_csv is not None:
        file_exists = os.path.exists(output_csv)
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Model", "V_CCC", "A_CCC", "D_CCC", "V_MSE", "A_MSE", "D_MSE", "V_MAE", "A_MAE", "D_MAE"])
            writer.writerow([
                os.path.basename(os.path.dirname(checkpoint_path)),
                valence_ccc, arousal_ccc, dominance_ccc,
                mse[0].item(), mse[1].item(), mse[2].item(),
                mae[0].item(), mae[1].item(), mae[2].item(),
            ])
        logging.info(f"Saved results to {output_csv}")

    # Save scatter plots
    if plot:
        vad_names = ["Valence", "Arousal", "Dominance"]
        y_true_np = y_true.cpu().numpy()
        y_pred = y_pred * 6 + 1
        y_pred_np = y_pred.cpu().numpy()
        os.makedirs("plots", exist_ok=True)

        for i, name in enumerate(vad_names):
            plt.figure()
            plt.scatter(y_true_np[:, 1], y_pred_np[:, 1], alpha=0.4)
            plt.xlabel(f"True {name}")
            plt.ylabel(f"Predicted {name}")
            plt.title(f"{name} Prediction vs Ground Truth\nCCC={ccc[i]:.3f}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"plots/{name.lower()}_scatter2.png")
            plt.close()
        logging.info("Saved scatter plots to ./plots/")

    return valence_ccc, arousal_ccc, dominance_ccc, mse, mae

def find_checkpoint_folder(path):
    candidate = os.listdir(path)
    if "logs" in candidate and "weights" in candidate and "cfg.log"in candidate:
        return [path]
    list_candidates = []
    for c in candidate:
        sub_path = os.path.join(path, c)
        if os.path.isdir(sub_path):
            list_candidates += find_checkpoint_folder(sub_path)
    return list_candidates

def main(args):
    logging.info("Finding checkpoints...")
    list_checkpoints = find_checkpoint_folder(args.checkpoint_path)
    test_set = args.test_set if args.test_set is not None else "test1.pkl"

    for ckpt in list_checkpoints:
        logging.info(f"Evaluating: {ckpt}")
        cfg_path = os.path.join(ckpt, "cfg.log")
        
        # Select checkpoint file
        if args.latest:
            ckpt_files = glob.glob(os.path.join(ckpt, "weights", "*.pt"))
            if len(ckpt_files) > 0:
                ckpt_path = ckpt_files[0]
                all_state_dict = True
            else:
                ckpt_path = glob.glob(os.path.join(ckpt, "weights", "*.pth"))[0]
                all_state_dict = False
        else:
            ckpt_path = os.path.join(ckpt, "weights/best_loss/checkpoint_0.pth")
            all_state_dict = False
        
        cfg = Config()
        cfg.load(cfg_path)
        cfg.data_valid = test_set
        if args.data_root is not None:
            assert args.data_name is not None, "Change validation dataset requires data_name"
            cfg.data_root = args.data_root
            cfg.data_name = args.data_name
        eval(cfg, ckpt_path, output_csv=args.output_csv, plot=args.plot, all_state_dict=all_state_dict)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ckpt", "--checkpoint_path", type=str, help="path to checkpoint folder")
    parser.add_argument("-r", "--recursive", action="store_true", help="whether to travel child folder or not")
    parser.add_argument("-l", "--latest", action="store_true", help="whether to use latest weight or best weight")
    parser.add_argument("-t", "--test_set", type=str, default=None, help="name of testing set. Ex: test.pkl")
    parser.add_argument("--data_root", type=str, default=None, help="If want to change the validation dataset")
    parser.add_argument("--data_name", type=str, default=None, help="for changing validation dataset")
    parser.add_argument("-o", "--output_csv", type=str, default="eval_result.csv", help="CSV file to store metrics")
    parser.add_argument("--plot", action="store_true", help="Save scatter plots of prediction vs ground truth for VAD")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    if args.recursive:
        main(args)
    else:
        # Single model evaluation
        cfg_path = os.path.join(args.checkpoint_path, "cfg.log")
        cfg = Config()
        cfg.load(cfg_path)
        test_set = args.test_set if args.test_set is not None else "test1.pkl"
        cfg.data_valid = test_set
        if args.data_root is not None:
            assert args.data_name is not None, "Changing dataset requires --data_name"
            cfg.data_root = args.data_root
            cfg.data_name = args.data_name

        # pick checkpoint
        ckpt_path = os.path.join(args.checkpoint_path, "weights/best_loss/checkpoint_0.pth")
        all_state_dict = False
        if not os.path.exists(ckpt_path):
            print(f"{ckpt_path} is not exist")
        eval(cfg, ckpt_path, output_csv=args.output_csv, plot=args.plot, all_state_dict=all_state_dict)