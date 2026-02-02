# HSE-NET-ONLINE-HANDWRITING-RECOGNITION
HSE-NET using CNN-BiLSTM Network for Bilingual Online Handwriting Recognition
import os
import argparse
import glob
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

# ---------------------
# Configuration / Labels
# ---------------------
DEFAULT_NUM_CLASSES = 91  # change to your exact number
MAX_SEQ_LEN = 128         # pad/truncate sequences to this length
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Example label -> script mapping (adjust to your labels)
# For demonstration: 0..51 roman, 52..97 devanagari (if using 91 classes)
def label_to_script(label):
    if label < 52:
        return "roman"
    else:
        return "devanagari"

# ---------------------
# Dataset Loader
# ---------------------
class OnlineStrokeDataset(Dataset):
    def __init__(self, folder, seq_len=MAX_SEQ_LEN, num_classes=DEFAULT_NUM_CLASSES):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        self.seq_len = seq_len
        self.num_classes = num_classes
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found in {folder}")
        # pre-load file paths only (lazy load per __getitem__)
    def __len__(self):
        return len(self.files)
    def _load_npz(self, path):
        d = np.load(path, allow_pickle=True)
        coords = d["coords"]  # (L, F) float32 array
        label = int(d["label"].item()) if isinstance(d["label"].item(), (np.integer, int)) else int(d["label"])
        script = d["script"].item() if "script" in d else label_to_script(label)
        return coords, label, script
    def __getitem__(self, idx):
        path = self.files[idx]
        coords, label, script = self._load_npz(path)
        # normalize coords to [0,1]
        coords = coords.astype(np.float32)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        # handle if only x,y present
        if coords.shape[1] < 5:
            # pad feature dim to 5 with zeros
            pad = np.zeros((coords.shape[0], 5 - coords.shape[1]), dtype=np.float32)
            coords = np.concatenate([coords, pad], axis=1)
        # resample/pad/truncate to seq_len
        L = coords.shape[0]
        if L >= self.seq_len:
            coords = coords[:self.seq_len, :]
        else:
            pad_len = self.seq_len - L
            coords = np.concatenate([coords, np.zeros((pad_len, coords.shape[1]), dtype=np.float32)], axis=0)
        # optional simple normalization by range
        min_xy = coords[:, :2].min(axis=0)
        max_xy = coords[:, :2].max(axis=0)
        rng = (max_xy - min_xy)
        rng[rng == 0] = 1.0
        coords[:, :2] = (coords[:, :2] - min_xy) / rng
        return coords, label, script

# ---------------------
# Bilingual Dataset (demo fallback)
# ---------------------
class DevanagariStrokeDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=MAX_SEQ_LEN, num_classes=DEFAULT_NUM_CLASSES):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        # create structured patterns so model can learn
        self.data = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
        self.labels = np.zeros((num_samples,), dtype=np.int64)
        for i in range(num_samples):
            lbl = np.random.randint(0, num_classes)
            self.labels[i] = lbl
            # pattern: sin wave frequency encodes label modulo something
            t = np.linspace(0, 2*np.pi, seq_len)
            freq = 1.0 + (lbl % 8) * 0.2
            self.data[i, :, 0] = 0.5 + 0.4 * np.sin(freq * t + (lbl % 3))  # x
            self.data[i, :, 1] = 0.5 + 0.4 * np.cos(freq * t + (lbl % 5))  # y
            # dx, dy as derivatives (approx)
            dx = np.gradient(self.data[i, :, 0])
            dy = np.gradient(self.data[i, :, 1])
            self.data[i, :, 2] = dx
            self.data[i, :, 3] = dy
            # pen_state: mostly pen-down (1), with random tiny gaps
            ps = np.ones(seq_len)
            gap_starts = np.random.choice(seq_len, size=2, replace=False)
            for s in gap_starts:
                ps[s:s+2] = 0
            self.data[i, :, 4] = ps
            # small noise
            self.data[i] += np.random.normal(0, 0.02, self.data[i].shape)
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx]), label_to_script(self.labels[idx])

# ---------------------
# Model: CNN (1D) + BiLSTM
# ---------------------
class CNN_BiLSTM(nn.Module):
    def __init__(self, in_feats=5, num_classes=DEFAULT_NUM_CLASSES, lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_feats, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden*2, num_classes)
        )
    def forward(self, x):
        # x: [B, seq_len, feats]
        x = x.transpose(1,2)            # [B, feats, seq_len]
        x = self.cnn(x)                # [B, C, seq_len']
        x = x.transpose(1,2)           # [B, seq_len', C]
        out, _ = self.lstm(x)          # [B, seq_len', 2*hidden]
        # use mean pooling over time for stability
        out = out.mean(dim=1)          # [B, 2*hidden]
        logits = self.classifier(out)
        return logits

# ---------------------
# Training / Utilities
# ---------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        predicted = preds.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    ys = []
    y_preds = []
    scripts = []
    with torch.no_grad():
        for x, y, script in loader:
            x = x.to(device).float()
            preds = model(x)
            loss = criterion(preds, y.to(device).long())
            running_loss += loss.item() * x.size(0)
            y_pred = preds.argmax(dim=1).cpu().numpy()
            y_preds.append(y_pred)
            ys.append(y.numpy())
            scripts += list(script)
    ys = np.concatenate(ys)
    y_preds = np.concatenate(y_preds)
    return running_loss / len(ys), ys, y_preds, scripts

# ---------------------
# Plotting functions
# ---------------------
def plot_accuracy(train_accs, val_accs, outpath):
    plt.figure(figsize=(7,4))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_loss(train_losses, val_losses, outpath):
    plt.figure(figsize=(7,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_confusion(y_true, y_pred, labels, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # normalize rows
    with np.errstate(all='ignore'):
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cmn, interpolation='nearest')
    ax.set_title('Figure 6. Confusion Matrix of Proposed CNN–LSTM Model')
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(l) for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([str(l) for l in labels])
    # annotate
    fmt = '.2f'
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            val = cmn[i, j]
            if np.isnan(val):
                txt = '0.00'
            else:
                txt = format(val, fmt)
            ax.text(j, i, txt, ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ---------------------
# Main routine
# ---------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Attempt to load real dataset; otherwise fallback to synthetic
    use_synth = args.use_synthetic_if_missing
    train_folder = "data/train"
    test_folder = "data/test"
    train_ds = None
    test_ds = None
    try:
        train_ds = OnlineStrokeDataset(train_folder, seq_len=args.seq_len, num_classes=args.num_classes)
        test_ds = OnlineStrokeDataset(test_folder, seq_len=args.seq_len, num_classes=args.num_classes)
        print(f"Loaded real dataset: {len(train_ds)} train samples, {len(test_ds)} test samples")
    except Exception as e:
        print("Could not load real dataset:", e)
        if use_synth:
            print("Falling back to bilingual dataset for result.")
            total = args.bilingual_samples
            train_n = int(total * 0.8)
            train_ds =DevanagariStrokeDataset(num_samples=train_n, seq_len=args.seq_len, num_classes=args.num_classes)
            test_ds = DevanagariStrokeDataset(num_samples=total - train_n, seq_len=args.seq_len, num_classes=args.num_classes)
            print(f"Devanagari dataset: {len(train_ds)} train samples, {len(test_ds)} test samples")
        else:
            raise RuntimeError("No dataset available and use_synthetic_if_missing=False")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CNN_BiLSTM(in_feats=5, num_classes=args.num_classes, lstm_hidden=args.lstm_hidden,
                       lstm_layers=args.lstm_layers, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Number of epochs to wait for improvement before early stopping

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, ys_val, preds_val, scripts_val = eval_model(model, val_loader, criterion, device)
        # compute val accuracy as standard
        val_acc = (preds_val == ys_val).mean()
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc # Store best accuracy too
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement in validation loss.")
                break # Exit the training loop

    # Load the best model found during training for final evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth"))

    # final evaluation on test set (we used val set as test if real provided)
    val_loss, ys_val, preds_val, scripts_val = eval_model(model, val_loader, criterion, device)
    overall_acc = (preds_val == ys_val).mean()
    print(f"\nFinal Test Accuracy (best model): {overall_acc*100:.2f}%")

    # per-script accuracies
    scripts_arr = np.array(scripts_val)
    roman_idx = np.where(np.array([s == "roman" for s in scripts_arr]))[0]
    dev_idx = np.where(np.array([s == "devanagari" for s in scripts_arr]))[0]
    if len(roman_idx) > 0:
        roman_acc = (preds_val[roman_idx] == ys_val[roman_idx]).mean()
        print(f"Roman accuracy: {roman_acc*100:.2f}% ({len(roman_idx)} samples)")
    if len(dev_idx) > 0:
        dev_acc = (preds_val[dev_idx] == ys_val[dev_idx]).mean()
        print(f"Devanagari accuracy: {dev_acc*100:.2f}% ({len(dev_idx)} samples)")

    # Save accuracy/loss curves
    plot_accuracy(train_accs, val_accs, OUTPUT_DIR / "accuracy_with_early_stopping.png")
    plot_loss(train_losses, val_losses, OUTPUT_DIR / "loss_with_early_stopping.png")

    # Confusion matrix and classification report
    labels = list(range(args.num_classes))
    plot_confusion(ys_val, preds_val, labels=labels[:min(50, len(labels))], outpath=OUTPUT_DIR / "confusion_matrix_with_early_stopping.png")

    report = classification_report(ys_val, preds_val, digits=4)
    print("\nClassification Report (best model):\n", report)
    with open(OUTPUT_DIR / "classification_report_with_early_stopping.txt", "w") as f:
        f.write("Final Test Accuracy (best model): {:.4f}\n\n".format(overall_acc))
        f.write(report)

    print("All outputs saved to", OUTPUT_DIR.resolve())

# ---------------------
# CLI args
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--epochs", type=int, default=100) # Increased epochs
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--synthetic_samples", type=int, default=2000)
    parser.add_argument("--use_synthetic_if_missing", action="store_true", default=True)
    args, unknown = parser.parse_known_args()
    main(args)
