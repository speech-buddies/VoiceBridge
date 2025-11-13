import os
import time
import torch
import whisper

# --- config ---
dataset_root = "trimmed_all_wavs"
model_name = "small"

# --- collect wav files from a flat folder ---
audio_files = [
    os.path.join(dataset_root, f)
    for f in os.listdir(dataset_root)
    if f.lower().endswith(".wav")
]

print(f"Found {len(audio_files)} wav files.")

# --- load model on GPU if available ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = whisper.load_model(model_name, device=device)

# --- time the full run ---
total_start = time.perf_counter()
per_file_times = []

for i, path in enumerate(audio_files, 1):
    print(f"[{i}/{len(audio_files)}] Processing: {path}")
    t0 = time.perf_counter()

    _ = model.transcribe(path)  # measure only

    dt = time.perf_counter() - t0
    per_file_times.append(dt)
    print(f"  took {dt:.2f} seconds")

total_time = time.perf_counter() - total_start

print("\n==== Timing summary ====")
print(f"Total wall-clock time : {total_time:.2f} seconds")
print(f"Average per file      : {sum(per_file_times)/len(per_file_times):.2f} seconds")
