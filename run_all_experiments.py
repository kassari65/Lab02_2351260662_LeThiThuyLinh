import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# Constants
FS = 16000
FRAME_MS = 25
HOP_MS = 10
WIN = int(FS * FRAME_MS / 1000) # 400
HOP = int(FS * HOP_MS / 1000)   # 160
LABELS = ['khong', 'mot', 'hai', 'ba', 'bon']

os.makedirs('figures', exist_ok=True)

def load_audio(path):
    y, sr = librosa.load(path, sr=FS, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def compute_time_features(y):
    num_frames = 1 + int((len(y) - WIN) / HOP)
    energy = np.zeros(num_frames)
    rms = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    w = np.hamming(WIN)
    
    for i in range(num_frames):
        start = i * HOP
        frame = y[start:start+WIN]
        frame_w = frame * w
        energy[i] = np.sum(frame_w ** 2)
        rms[i] = np.sqrt(np.mean(frame_w ** 2))
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr[i] = np.sum(np.abs(signs[1:] - signs[:-1])) / (2.0 * WIN)
        
    log_energy = 10.0 * np.log10(energy + 1e-12)
    times = (np.arange(num_frames) * HOP + WIN / 2) / FS
    return times, energy, log_energy, zcr

def compute_autocorrelation_pitch(y):
    # take frame in middle of voiced portion
    frame_idx = len(y) // (2 * HOP)
    start = frame_idx * HOP
    frame = y[start:start+WIN] * np.hamming(WIN)
    r = np.correlate(frame, frame, mode='full')
    r = r[len(r)//2:]
    
    min_lag = int(FS / 400) # 40
    max_lag = int(FS / 60)  # ~266
    peak_lag = min_lag + np.argmax(r[min_lag:max_lag])
    f0 = FS / peak_lag if peak_lag > 0 else 0
    return r, peak_lag, f0

def trim_energy(y, top_db=30, margin_ms=50):
    yt, idx = librosa.effects.trim(y, top_db=top_db, frame_length=WIN, hop_length=HOP)
    m = int(FS * margin_ms / 1000)
    s = max(0, idx[0] - m)
    e = min(len(y), idx[1] + m)
    return y[s:e], (s, e)

def mfcc_feature(y, n_mfcc=13, n_mels=24, use_delta=False):
    y_pre = lfilter([1.0, -0.97], [1.0], y)
    M = librosa.feature.mfcc(
        y=y_pre, sr=FS, n_mfcc=n_mfcc, n_mels=n_mels,
        n_fft=512, win_length=WIN, hop_length=HOP,
        window='hamming', center=False
    )
    # CMN
    M = M - np.mean(M, axis=1, keepdims=True)
    if use_delta:
        delta = librosa.feature.delta(M)
        M = np.vstack([M, delta])
    return M.T

def dtw_distance(X, Y):
    N, M = len(X), len(Y)
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    C = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((N + 1, M + 1, 2), dtype=int)
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c_cost = C[i - 1, j - 1]
            d_up = D[i - 1, j]
            d_left = D[i, j - 1]
            d_diag = D[i - 1, j - 1]
            
            if d_diag <= d_up and d_diag <= d_left:
                best = d_diag
                pi, pj = i - 1, j - 1
            elif d_up <= d_left:
                best = d_up
                pi, pj = i - 1, j
            else:
                best = d_left
                pi, pj = i, j - 1
                
            D[i, j] = c_cost + best
            back[i, j] = (pi, pj)
            
    path = []
    i, j = N, M
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = back[i, j]
    path.reverse()
    
    dtw_norm = D[N, M] / max(len(path), 1)
    return dtw_norm, path, D[1:, 1:], C

def build_templates(root='dataset', trim=True, use_delta=False, num_templates=3):
    templates = {lab: [] for lab in LABELS}
    for lab in LABELS:
        files = sorted(Path(root, lab).glob('*.wav'))[:num_templates]
        for f in files:
            y = load_audio(f)
            if trim:
                y, _ = trim_energy(y)
            feat = mfcc_feature(y, use_delta=use_delta)
            templates[lab].append(feat)
    return templates

def recognize(path, templates, trim=True, use_delta=False):
    y = load_audio(path)
    if trim:
        y, _ = trim_energy(y)
    X = mfcc_feature(y, use_delta=use_delta)
    scores = {}
    for lab, refs in templates.items():
        scores[lab] = min(dtw_distance(X, R)[0] for R in refs)
    pred = min(scores, key=scores.get)
    sorted_scores = dict(sorted(scores.items(), key=lambda kv: kv[1]))
    return pred, sorted_scores

# Execution & Figure Generation
print("1. Generating Waveform Plots (figures/fig1_waveforms.png)...")
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
sample_words = ['khong', 'hai', 'bon']
for idx, w in enumerate(sample_words):
    y = load_audio(f"dataset/{w}/{w}_01.wav")
    t = np.arange(len(y)) / FS
    axes[idx].plot(t, y, color='#1f77b4', lw=1.2)
    axes[idx].set_title(f"Waveform từ '{w}' (khong_01.wav) - Mono 16 kHz", fontsize=11, fontweight='bold')
    axes[idx].set_ylabel("Biên độ")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim(-1.1, 1.1)
axes[-1].set_xlabel("Thời gian (giây)")
plt.tight_layout()
plt.savefig("figures/fig1_waveforms.png", dpi=300)
plt.close()

print("2. Generating Time Features: Waveform, Log-Energy, ZCR (figures/fig2_time_features.png)...")
fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex='col')
for col, w in enumerate(['khong', 'mot', 'hai']):
    y = load_audio(f"dataset/{w}/{w}_01.wav")
    t = np.arange(len(y)) / FS
    t_feat, energy, log_e, zcr = compute_time_features(y)
    
    # Row 0: Waveform
    axes[0, col].plot(t, y, color='#2ca02c', lw=1.0)
    axes[0, col].set_title(f"Từ: '{w}'", fontsize=12, fontweight='bold')
    axes[0, col].set_ylabel("Biên độ")
    axes[0, col].grid(True, alpha=0.3)
    
    # Row 1: Log-energy
    axes[1, col].plot(t_feat, log_e, color='#d62728', lw=1.5)
    axes[1, col].set_ylabel("Log-Energy (dB)")
    axes[1, col].grid(True, alpha=0.3)
    
    # Row 2: ZCR
    axes[2, col].plot(t_feat, zcr, color='#9467bd', lw=1.5)
    axes[2, col].set_ylabel("Zero-Crossing Rate")
    axes[2, col].set_xlabel("Thời gian (s)")
    axes[2, col].grid(True, alpha=0.3)
plt.suptitle("Phân tích miền thời gian: Waveform, Log-Energy và ZCR", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig2_time_features.png", dpi=300)
plt.close()

print("3. Generating Pitch & Autocorrelation Plot (figures/fig3_autocorrelation.png)...")
y = load_audio("dataset/ba/ba_01.wav")
r, lag, f0 = compute_autocorrelation_pitch(y)
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(r[:400])) / FS * 1000, r[:400], color='#1f77b4', lw=1.5)
plt.axvline(lag / FS * 1000, color='r', linestyle='--', label=f'Đỉnh pitch N0 = {lag} mẫu (~{lag/FS*1000:.1f} ms, F0 ≈ {f0:.1f} Hz)')
plt.title(f"Hàm tự tương quan ngắn hạn (Short-time Autocorrelation) của âm '/a/' trong từ 'ba'", fontsize=12, fontweight='bold')
plt.xlabel("Độ trễ lag (ms)")
plt.ylabel("R(k)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3_autocorrelation.png", dpi=300)
plt.close()

print("4. Generating Endpoint Detection Trimming (figures/fig4_endpoint_detection.png)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 6))
words_test = ['khong', 'mot']
for idx, w in enumerate(words_test):
    y_raw = load_audio(f"dataset/{w}/{w}_01.wav")
    y_trim, (s, e) = trim_energy(y_raw, top_db=30, margin_ms=50)
    t_raw = np.arange(len(y_raw)) / FS
    t_trim = np.arange(len(y_trim)) / FS
    
    axes[idx, 0].plot(t_raw, y_raw, color='#7f7f7f', lw=1.0)
    axes[idx, 0].axvline(s/FS, color='g', linestyle='--', label=f'Start ({s/FS:.2f}s)')
    axes[idx, 0].axvline(e/FS, color='r', linestyle='--', label=f'End ({e/FS:.2f}s)')
    axes[idx, 0].set_title(f"Gốc '{w}_01.wav' (Độ dài: {len(y_raw)/FS:.2f}s)", fontsize=11, fontweight='bold')
    axes[idx, 0].set_ylabel("Biên độ")
    axes[idx, 0].legend(loc='upper right')
    axes[idx, 0].grid(True, alpha=0.3)
    
    axes[idx, 1].plot(t_trim, y_trim, color='#2ca02c', lw=1.2)
    axes[idx, 1].set_title(f"Sau trim '{w}_01.wav' (Độ dài: {len(y_trim)/FS:.2f}s)", fontsize=11, fontweight='bold')
    axes[idx, 1].grid(True, alpha=0.3)
axes[1, 0].set_xlabel("Thời gian (s)")
axes[1, 1].set_xlabel("Thời gian (s)")
plt.suptitle("Minh họa Endpoint Detection (Cắt tỉa khoảng lặng)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig4_endpoint_detection.png", dpi=300)
plt.close()

print("5. Generating Mel Filterbank & MFCC Heatmap (figures/fig5_mfcc_heatmap.png)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y_khong = load_audio("dataset/khong/khong_01.wav")
y_khong_t, _ = trim_energy(y_khong)
mfcc_khong = mfcc_feature(y_khong_t)

y_hai = load_audio("dataset/hai/hai_01.wav")
y_hai_t, _ = trim_energy(y_hai)
mfcc_hai = mfcc_feature(y_hai_t)

im0 = axes[0].imshow(mfcc_khong.T, origin='lower', aspect='auto', cmap='magma')
axes[0].set_title(f"MFCC của từ 'khong' ({mfcc_khong.shape[0]} frames x {mfcc_khong.shape[1]} coeff)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Chỉ số MFCC (0-12)")
axes[0].set_xlabel("Chỉ số Frame (10 ms/frame)")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(mfcc_hai.T, origin='lower', aspect='auto', cmap='magma')
axes[1].set_title(f"MFCC của từ 'hai' ({mfcc_hai.shape[0]} frames x {mfcc_hai.shape[1]} coeff)", fontsize=11, fontweight='bold')
axes[1].set_ylabel("Chỉ số MFCC (0-12)")
axes[1].set_xlabel("Chỉ số Frame (10 ms/frame)")
fig.colorbar(im1, ax=axes[1])
plt.suptitle("Đặc trưng MFCC (Mel-Frequency Cepstral Coefficients) sau CMN", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig5_mfcc_heatmap.png", dpi=300)
plt.close()

print("6. Generating DTW Alignment Path Plots (figures/fig6_dtw_paths.png)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Intra-word: khong_01 vs khong_02
y_k1 = trim_energy(load_audio("dataset/khong/khong_01.wav"))[0]
y_k2 = trim_energy(load_audio("dataset/khong/khong_02.wav"))[0]
feat_k1 = mfcc_feature(y_k1)
feat_k2 = mfcc_feature(y_k2)
cost_same, path_same, D_same, C_same = dtw_distance(feat_k1, feat_k2)
px_same = [p[0] for p in path_same]
py_same = [p[1] for p in path_same]

im0 = axes[0].imshow(C_same, origin='lower', aspect='auto', cmap='viridis')
axes[0].plot(py_same, px_same, color='r', lw=2.5, label='Warping path')
axes[0].set_title(f"Cùng từ: 'khong_01' vs 'khong_02'\nDTW_norm = {cost_same:.3f}", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Frame 'khong_02'")
axes[0].set_ylabel("Frame 'khong_01'")
axes[0].legend(loc='upper left')
fig.colorbar(im0, ax=axes[0], label='Khoảng cách Euclid')

# Inter-word: khong_01 vs hai_01
y_h1 = trim_energy(load_audio("dataset/hai/hai_01.wav"))[0]
feat_h1 = mfcc_feature(y_h1)
cost_diff, path_diff, D_diff, C_diff = dtw_distance(feat_k1, feat_h1)
px_diff = [p[0] for p in path_diff]
py_diff = [p[1] for p in path_diff]

im1 = axes[1].imshow(C_diff, origin='lower', aspect='auto', cmap='viridis')
axes[1].plot(py_diff, px_diff, color='r', lw=2.5, label='Warping path')
axes[1].set_title(f"Khác từ: 'khong_01' vs 'hai_01'\nDTW_norm = {cost_diff:.3f}", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Frame 'hai_01'")
axes[1].set_ylabel("Frame 'khong_01'")
axes[1].legend(loc='upper left')
fig.colorbar(im1, ax=axes[1], label='Khoảng cách Euclid')

plt.suptitle("Ma trận khoảng cách cục bộ C và đường căn chỉnh DTW tối ưu", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/fig6_dtw_paths.png", dpi=300)
plt.close()

print("7. Running Full Evaluation and Generating results.csv & Confusion Matrix...")
templates = build_templates('dataset', trim=True, use_delta=False)
records = []
y_true = []
y_pred = []

for lab in LABELS:
    test_files = sorted((Path('dataset')/lab).glob('*.wav'))[3:]
    for f in test_files:
        pred, scores = recognize(f, templates, trim=True, use_delta=False)
        items = list(scores.items())
        y_true.append(lab)
        y_pred.append(pred)
        records.append({
            'file_name': f.name,
            'true_label': lab,
            'pred_label': pred,
            'top1_label': items[0][0],
            'top1_score': round(items[0][1], 4),
            'top2_label': items[1][0],
            'top2_score': round(items[1][1], 4)
        })

df_results = pd.DataFrame(records)
df_results.to_csv("results.csv", index=False)
print("Saved results.csv successfully")

cm = confusion_matrix(y_true, y_pred, labels=LABELS)
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc*100:.1f}%")

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS, cbar=False)
plt.title(f"Ma trận nhầm lẫn (Confusion Matrix) - Accuracy = {acc*100:.1f}%", fontsize=12, fontweight='bold')
plt.xlabel("Nhãn dự đoán")
plt.ylabel("Nhãn thực tế")
plt.tight_layout()
plt.savefig("figures/fig7_confusion_matrix.png", dpi=300)
plt.close()

# Experiment E1: Trim vs No-Trim
print("8. Running Experiment E1: Trim vs No-Trim...")
tpl_notrim = build_templates('dataset', trim=False, use_delta=False)
e1_preds_trim, e1_preds_notrim = [], []
for lab in LABELS:
    test_files = sorted((Path('dataset')/lab).glob('*.wav'))[3:]
    for f in test_files:
        p_trim, _ = recognize(f, templates, trim=True, use_delta=False)
        p_notrim, _ = recognize(f, tpl_notrim, trim=False, use_delta=False)
        e1_preds_trim.append(p_trim)
        e1_preds_notrim.append(p_notrim)

acc_trim = accuracy_score(y_true, e1_preds_trim)
acc_notrim = accuracy_score(y_true, e1_preds_notrim)
print(f"E1 Results: Trim Acc = {acc_trim*100:.1f}%, No-Trim Acc = {acc_notrim*100:.1f}%")

# Experiment E2: 13 MFCC vs 13 MFCC + Delta (26 dims)
print("9. Running Experiment E2: 13 MFCC vs 13 MFCC + Delta...")
tpl_delta = build_templates('dataset', trim=True, use_delta=True)
e2_preds_delta = []
for lab in LABELS:
    test_files = sorted((Path('dataset')/lab).glob('*.wav'))[3:]
    for f in test_files:
        p_d, _ = recognize(f, tpl_delta, trim=True, use_delta=True)
        e2_preds_delta.append(p_d)

acc_delta = accuracy_score(y_true, e2_preds_delta)
print(f"E2 Results: MFCC 13 Acc = {acc_trim*100:.1f}%, MFCC + Delta Acc = {acc_delta*100:.1f}%")

print("All pipeline steps and figures generated successfully!")
