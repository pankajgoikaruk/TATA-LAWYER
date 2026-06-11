import numpy as np, librosa


def bandpass(y, sr, lo, hi):
    # simple butter-like via FFT mask (ok for baseline)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    mask = (freqs >= lo) & (freqs <= hi)
    return np.fft.irfft(Y * mask.astype(float), n=len(y))


def to_logmel(y, sr, n_mels=64, n_fft=1024, win_ms=25, hop_ms=10):
    hop = int(sr*hop_ms/1000)
    win = int(sr*win_ms/1000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
    win_length=win, n_mels=n_mels, power=2.0)
    S = librosa.power_to_db(S, ref=np.max)
    return S.astype(np.float32)


def cmvn_feat(S):
    mu = S.mean(axis=1, keepdims=True)
    std = S.std(axis=1, keepdims=True) + 1e-8
    return ((S - mu)/std).astype(np.float32)