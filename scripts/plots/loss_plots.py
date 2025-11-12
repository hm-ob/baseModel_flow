import re
import matplotlib.pyplot as plt
import os

# ログファイルを読み込む
log_path = "/workspace/mount/scripts/working/checkpoints/MSP-Podcast/MemoCMT_bert_hubert_base/20251111-171543/logs/20251111-171544/training.log"  # ←ログのパスに合わせて変更
date = "20251111-171543"
save_path = "/workspace/mount/scripts/plots/{}/loss_curve.png".format(date)
save_path1 = "/workspace/mount/scripts/plots/{}/validation_CCCs_curve.png".format(date)
#save_path2 = "/workspace/mount/scripts/plots/{}/arousalCCC_curve.png".format(date)
#save_path3 = "/workspace/mount/scripts/plots/{}/dominanceCCC_curve.png".format(date)
with open(log_path, "r") as f:
    logs = f.readlines()

# 記録用辞書
epochs = []
train_loss, val_loss = [], []
train_valence_ccc, train_arousal_ccc, train_dominance_ccc, val_valence_ccc, val_arousal_ccc, val_dominance_ccc = [], [], [], [], [], []
sum_ccc, mean_ccc = [], []

if not os.path.exists("/workspace/mount/scripts/plots/{}".format(date)):
    os.makedirs("/workspace/mount/scripts/plots/{}".format(date))

# 解析
for line in logs:
    # Epoch番号
    epoch_match = re.search(r"Epoch (\d+)/\d+", line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        if current_epoch not in epochs:
            epochs.append(current_epoch)

    # Train loss
    if "loss:" in line and "Validation" not in line:
        loss_match = re.search(r"loss:\s*([0-9.]+)", line)
        if loss_match:
            train_loss.append(float(loss_match.group(1)))

    # Validation loss
    if "Validation: loss:" in line:
        val_loss_match = re.search(r"loss:\s*([0-9.]+)", line)
        if val_loss_match:
            val_loss.append(float(val_loss_match.group(1)))

    # Validation CCC
    if "Validation v_ccc:" in line:
        v_ccc_match = re.search(r"v_ccc:\s*([0-9.]+)", line)
        if v_ccc_match:
            val_valence_ccc.append(float(v_ccc_match.group(1)))

    if "Validation a_ccc:" in line:
        a_ccc_match = re.search(r"a_ccc:\s*([0-9.]+)", line)
        if a_ccc_match:
            val_arousal_ccc.append(float(a_ccc_match.group(1)))

    if "Validation d_ccc:" in line:
        d_ccc_match = re.search(r"d_ccc:\s*([0-9.]+)", line)
        if d_ccc_match:
            val_dominance_ccc.append(float(d_ccc_match.group(1)))


for v in range(len(val_valence_ccc)):
    sum_ccc.append(val_valence_ccc[v] + val_arousal_ccc[v] + val_dominance_ccc[v])
    mean_ccc.append(sum_ccc[v] / 3)

# グラフ描画
plt.figure(figsize=(8,5))
plt.plot(epochs[:len(train_loss)], train_loss, label='Train Loss', marker='o')
plt.plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(save_path) 
#plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs[:len(val_valence_ccc)], val_valence_ccc, label='Valence CCC', marker='o')
plt.plot(epochs[:len(val_arousal_ccc)], val_arousal_ccc, label='Arousal CCC', marker='o')
plt.plot(epochs[:len(val_dominance_ccc)], val_dominance_ccc, label='Dominance CCC', marker='o')
plt.plot(epochs[:len(mean_ccc)], mean_ccc, label='Mean CCC', marker='o')
plt.xlabel("Epoch")
plt.ylabel("CCC")
plt.title("Validation CCCs Curve")
plt.legend()
plt.grid(True)
plt.savefig(save_path1) 
#plt.show()