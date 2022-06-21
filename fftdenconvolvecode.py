import math
import numpy as np
import scipy.io.wavfile as wavfile

IR_PATH = "case1_ir.wav"
AUDIO_EXCERPT_PATH = "case1_gtr.wav"

# %% Loading the IRs

fs, case_ir_sig = wavfile.read(IR_PATH)
case_ir_sig = case_ir_sig[case_ir_sig != 0]

rec_fs, rec_sig = wavfile.read(AUDIO_EXCERPT_PATH)
assert (fs == rec_fs)

y = rec_sig
h = case_ir_sig

lenx = len(rec_sig)
lenh = len(case_ir_sig)

L = lenx + lenh - 1
threshold = -30
regu = 1e-5

Y = np.fft.fft(y, L)
H = np.fft.fft(h, L)
lam = [0 if 20 * math.log10(abs(i)) > threshold else regu for i in H]
X = Y / (H + lam)
x = np.fft.ifft(X, L)
x = x[:lenx]

dry_sig = x.astype(rec_sig.dtype)
dry_sig = dry_sig / np.max(np.abs(dry_sig))

wavfile.write("case11.wav", fs, dry_sig)