from pydub import AudioSegment, silence
from typing import Optional, List, Tuple
import os

from pathlib import Path
from typing import Optional, List, Tuple
import tarfile, shutil, json, sys


def remove_long_pauses(
    input_wav: str,
    output_wav: str,
    pause_length: float,
    *,
    silence_thresh_db: Optional[int] = None,
    edge_padding_ms: int = 30,
    crossfade_ms: int = 20,
) -> str:
    """
    Remove silences (pauses) longer than `pause_length` seconds from a WAV file.
    """
    if pause_length <= 0:
        raise ValueError("pause_length must be > 0 seconds")

    audio = AudioSegment.from_file(input_wav)
    if silence_thresh_db is None:
        silence_thresh_db = int(audio.dBFS - 16)

    min_silence_len_ms = int(pause_length * 1000)

    # Find only the silent intervals that are at least the threshold long
    silent_ranges: List[Tuple[int, int]] = silence.detect_silence(
        audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db
    )
    # Early exit: nothing to remove
    if not silent_ranges:
        audio.export(output_wav, format="wav")
        return output_wav

    # Build "keep" ranges by cutting out long silences.
    keeps: List[AudioSegment] = []
    prev_end = 0
    n = len(audio)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    for start, end in silent_ranges:
        keep_start = prev_end
        keep_end = clamp(start + edge_padding_ms, 0, n)
        if keep_end > keep_start:
            keeps.append(audio[keep_start:keep_end])
        prev_end = clamp(end - edge_padding_ms, 0, n)

    # Remainder after the last silence
    if prev_end < n:
        keeps.append(audio[prev_end:n])

    # Stitch together with a small crossfade for smoothness
    if not keeps:
        AudioSegment.silent(duration=100, frame_rate=audio.frame_rate).export(
            output_wav, format="wav"
        )
        return output_wav

    out = keeps[0]
    for seg in keeps[1:]:
        out = out.append(seg, crossfade=crossfade_ms if crossfade_ms > 0 else 0)

    out.export(output_wav, format="wav")
    return output_wav


# === CHANGED: input is "all_wavs" instead of tar-based dataset ===
INPUT_ROOT = Path("all_wavs")
# This will create a sibling folder called "trimmed_all_wavs"
OUTPUT_ROOT = INPUT_ROOT.with_name("trimmed_all_wavs")
PAUSE_LENGTH_SECONDS = 0.2  # adjust here if you ever need to


def _safe_extract_all(tar_obj: tarfile.TarFile, dest: Path) -> None:
    """(Unused now, kept just in case)"""
    dest = dest.resolve()
    for m in tar_obj.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest)):
            raise Exception(f"Blocked unsafe path in tar: {m.name}")
    tar_obj.extractall(dest)


def extract_tar_once(tar_path: Path) -> Path:
    """(Unused in the all_wavs version, kept for compatibility)"""
    out = tar_path.parent / f"{tar_path.stem}_extracted"
    if out.exists() and any(out.iterdir()):
        print(f"[=] Already extracted: {out}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"[+] Extracting: {tar_path} -> {out}/")
    with tarfile.open(tar_path, "r") as t:
        _safe_extract_all(t, out)
    return out


def process_tree(inner_root: Path, mirror_root: Path) -> tuple[int, int]:
    """
    Copy .json as-is and write cleaned .wav with the SAME filename (.wav) to mirror_root,
    preserving the relative structure under inner_root.
    Returns (json_copied, wav_cleaned).
    """
    json_copied = 0
    wav_cleaned = 0

    for p in inner_root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(inner_root)
        dst = mirror_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        ext = p.suffix.lower()
        if ext == ".json":
            shutil.copy2(p, dst)
            json_copied += 1
            print(f"[→] JSON : {rel}")
        elif ext == ".wav":
            try:
                remove_long_pauses(p, dst, pause_length=PAUSE_LENGTH_SECONDS)
                wav_cleaned += 1
                print(f"[·] WAV  : {rel} (cleaned)")
            except Exception as e:
                print(f"[x] Failed to clean {rel}: {e}")
        else:
            # ignore other file types
            pass

    return json_copied, wav_cleaned


def main():
    if not INPUT_ROOT.exists() or not INPUT_ROOT.is_dir():
        print(f"[!] Expected input folder not found: {INPUT_ROOT.resolve()}")
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Input : {INPUT_ROOT.resolve()}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")

    # === CHANGED: no tar loop, just process all_wavs directly ===
    total_json, total_wav = process_tree(INPUT_ROOT, OUTPUT_ROOT)

    print("\n=== Summary ===")
    print(f"JSON copied : {total_json}")
    print(f"WAV cleaned : {total_wav}")
    print(f"Trimmed dir : {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
