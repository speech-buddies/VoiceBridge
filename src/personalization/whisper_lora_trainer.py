"""
lora_trainer.py — Incremental LoRA fine-tuning for Whisper-small (CPU-compatible)
==================================================================================
Designed as a recurring job that continuously improves the model as new audio
data arrives in the target directory. On each run it:

  1. Loads the existing LoRA adapter if one has been saved previously, so
     training is cumulative rather than starting from scratch each time.
  2. Tracks which files have already been trained on in a seen-files ledger
     (seen_files.json inside output_dir). Each run trains on new files plus a
     random replay sample of previously-seen files, preventing catastrophic
     forgetting of older data while still prioritising new samples.
  3. Saves the updated adapter back to the same location when done, ready
     for the next run.

Typical recurring-job usage
----------------------------
    from lora_trainer import WhisperLoRATrainer

    trainer = WhisperLoRATrainer(
        data_dir   = "./my_audio_samples",
        meta_file  = "metadata.json",
        output_dir = "./checkpoints",
    )
    trainer.setup()           # resumes from existing adapter if present;
                              # filters out already-seen files automatically
    if trainer.has_new_data:
        trainer.train()
        trainer.evaluate()
        trainer.save()        # overwrites adapter in-place for the next run
    else:
        print("No new data since last run.")

Expected metadata JSON format (list of objects):
    [
        {
            "id": "53c9a174341845eb8da8dee0331aea83",
            "transcript": "Open the Wikipedia.",
            "wav_file": "53c9a174341845eb8da8dee0331aea83.wav",
            "duration_s": 3.27,
            "timestamp": "2026-03-11T15:54:29.640267+00:00"
        },
        ...
    ]
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, PeftModel

# Filename of the ledger that records which audio files have been trained on.
_SEEN_FILES_LEDGER = "seen_files.json"

# Subdirectory within output_dir where the live adapter is stored.
_ADAPTER_SUBDIR = "./models/adapters"


# ---------------------------------------------------------------------------
# Internal Dataset
# ---------------------------------------------------------------------------

class _AudioTextDataset(Dataset):
    """PyTorch Dataset that loads audio files and tokenises transcripts."""

    def __init__(self, data_list: List[Tuple[str, str]], processor: WhisperProcessor,
                 sampling_rate: int = 16_000):
        self.data_list = data_list
        self.processor = processor
        self.sr = sampling_rate

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, transcript = self.data_list[idx]

        # Load audio
        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Stereo → mono (handles both (samples, ch) and (ch, samples) layouts)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)

        # Resample if the file's native rate differs from the target
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)

        # Extract log-mel features expected by Whisper
        input_features = self.processor.feature_extractor(
            waveform.numpy(), sampling_rate=self.sr
        ).input_features[0]

        # Tokenise the ground-truth transcript
        labels = self.processor.tokenizer(
            transcript, add_special_tokens=True
        ).input_ids

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Internal Data Collator
# ---------------------------------------------------------------------------

@dataclass
class _DataCollator:
    """Pads a batch of variable-length features and label sequences."""
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels_batch   = [{"input_ids":       f["labels"]}         for f in features]

        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(labels_batch, return_tensors="pt")["input_ids"]

        # Mask padding positions so they don't contribute to the loss
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class WhisperLoRATrainer:
    """
    Incrementally fine-tunes Whisper-small with LoRA adapters on a custom
    audio dataset, designed to be called repeatedly as new data arrives.

    Only the decoder's attention projections are trained.

    Incremental behaviour
    ---------------------
    - On the first run, a fresh LoRA adapter is initialised and all available
      data is used for training.
    - On subsequent runs, the previously saved adapter is loaded. Training data
      consists of all new files plus a random replay sample drawn from the
      seen-files ledger. The replay sample size is ``replay_ratio`` × the number
      of new files, capped at however many old files actually exist. This keeps
      the model anchored to prior knowledge while still learning new patterns.
    - After save() is called, the ledger is updated to mark the current batch
      of files as seen, and the adapter is overwritten in-place.

    Parameters
    ----------
    data_dir : str
        Directory containing the audio files.
    meta_file : str
        Path to the JSON metadata file. If relative, resolved against data_dir.
    output_dir : str
        Directory for checkpoints, the live adapter, and the seen-files ledger.
    epochs : int
        Number of full passes over the *new* training data each run.
    batch_size : int
        Samples per batch. Keep at 2–4 for CPU.
    lr : float
        AdamW learning rate.
    test_split : float
        Fraction of *new* data held out for evaluation (0–1).
    seed : int
        Random seed for reproducible train/test splits.
    sampling_rate : int
        Target audio sampling rate (Whisper-small expects 16 000 Hz).
    print_every : int
        Print average loss every N batches.
    checkpoint_every : int
        Save a mid-run checkpoint every N global steps.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout applied inside LoRA layers.
    replay_ratio : float
        Controls how many old samples are replayed relative to the number of
        new samples. A value of 0.5 means the replay buffer will contain up to
        50% as many old samples as there are new ones. Set to 0.0 to disable
        replay entirely. Defaults to 0.5.
    on_batch_end : Optional[Callable]
        Optional callback after every batch: ``fn(epoch, batch_idx, step, loss)``.
    on_epoch_end : Optional[Callable]
        Optional callback after every epoch: ``fn(epoch, avg_train_loss)``.
    """

    def __init__(
        self,
        data_dir:         str,
        meta_file:        str   = "metadata.json",
        output_dir:       str   = "./checkpoints",
        epochs:           int   = 3,
        batch_size:       int   = 2,
        lr:               float = 1e-4,
        test_split:       float = 0.1,
        seed:             int   = 42,
        sampling_rate:    int   = 16_000,
        print_every:      int   = 10,
        checkpoint_every: int   = 100,
        lora_r:           int   = 16,
        lora_alpha:       int   = 32,
        lora_dropout:     float = 0.1,
        replay_ratio:     float = 0.5,
        on_batch_end:     Optional[Callable] = None,
        on_epoch_end:     Optional[Callable] = None,
    ):
        self.data_dir         = data_dir
        self.meta_file        = meta_file
        self.output_dir       = output_dir
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.lr               = lr
        self.test_split       = test_split
        self.seed             = seed
        self.sampling_rate    = sampling_rate
        self.print_every      = print_every
        self.checkpoint_every = checkpoint_every
        self.lora_r           = lora_r
        self.lora_alpha       = lora_alpha
        self.lora_dropout     = lora_dropout
        self.replay_ratio     = replay_ratio
        self.on_batch_end     = on_batch_end
        self.on_epoch_end     = on_epoch_end

        # CPU-only
        self.device = "cpu"

        # Populated by setup()
        self.processor:    Optional[WhisperProcessor]      = None
        self.model:        Optional[PeftModel]             = None
        self.optimizer:    Optional[torch.optim.Optimizer] = None
        self.train_loader: Optional[DataLoader]            = None
        self.test_loader:  Optional[DataLoader]            = None

        # Filenames included in the current training run (populated by setup())
        self._current_run_files: List[str] = []

        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_new_data(self) -> bool:
        """True if setup() found unseen files to train on."""
        return len(self._current_run_files) > 0

    def setup(self) -> None:
        """
        Prepare for a training run.

        - Reads the seen-files ledger and filters metadata to only new samples.
        - Loads the existing LoRA adapter if one is present, otherwise
          initialises a fresh one.
        - Builds DataLoaders over the new samples only.

        Call this before train() on every scheduled run.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[WhisperLoRATrainer] Device: {self.device}")

        # ---- Filter to unseen files only ----
        all_data = self._load_data()
        seen     = self._load_seen_files()
        new_data = [(p, t) for p, t in all_data if os.path.basename(p) not in seen]
        old_data = [(p, t) for p, t in all_data if os.path.basename(p) in seen]

        total, new = len(all_data), len(new_data)
        print(f"[WhisperLoRATrainer] {total} total files | {total - new} already seen | {new} new")

        # ---- Experience replay: mix old samples in with the new data ----
        # A random subset of previously-seen files is replayed each run 
        # The replay buffer is sized as replay_ratio × len(new_data)
        replay_data = self._sample_replay_data(old_data, len(new_data))
        if replay_data:
            print(f"[WhisperLoRATrainer] Replaying {len(replay_data)} old samples "
                  f"alongside {len(new_data)} new ones (ratio={self.replay_ratio})")

        combined_data = new_data + replay_data

        # Record only the *new* filenames so replay samples aren't double-counted
        # in the ledger — they're already marked as seen from a prior run.
        self._current_run_files = [os.path.basename(p) for p, _ in new_data]

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model     = self._build_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if not new_data:
            print("[WhisperLoRATrainer] No new data — skipping DataLoader construction.")
            return

        train_data, test_data = self._split(combined_data)
        print(f"[WhisperLoRATrainer] New split — Train: {len(train_data)} | Test: {len(test_data)}")

        collator = _DataCollator(processor=self.processor)

        self.train_loader = DataLoader(
            _AudioTextDataset(train_data, self.processor, self.sampling_rate),
            batch_size=self.batch_size, shuffle=True,  collate_fn=collator,
        )
        self.test_loader = DataLoader(
            _AudioTextDataset(test_data, self.processor, self.sampling_rate),
            batch_size=self.batch_size, shuffle=False, collate_fn=collator,
        )

        print("[WhisperLoRATrainer] Setup complete.")

    def train(self) -> List[float]:
        """
        Train on the new data discovered during setup().

        Returns
        -------
        List[float]
            Average training loss per epoch.
        """
        self._require_setup()
        if not self.has_new_data:
            print("[WhisperLoRATrainer] Nothing to train on — no new files.")
            return []

        epoch_losses = []
        for epoch in range(1, self.epochs + 1):
            print(f"\n[WhisperLoRATrainer] === Epoch {epoch}/{self.epochs} ===")
            avg_loss = self._train_epoch(epoch)
            epoch_losses.append(avg_loss)

            ckpt_path = os.path.join(self.output_dir, f"whisper_lora_epoch{epoch}.pt")
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"[WhisperLoRATrainer] Epoch {epoch} done — loss {avg_loss:.4f} → {ckpt_path}")

            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_loss)

        return epoch_losses

    def evaluate(self) -> float:
        """
        Evaluate on the held-out portion of the new data.

        Returns
        -------
        float
            Average loss over the test set, or NaN if no test data is available.
        """
        self._require_setup()
        if self.test_loader is None:
            print("[WhisperLoRATrainer] No test data available.")
            return float("nan")

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader, start=1):
                input_features = batch["input_features"].to(self.device)
                labels         = batch["labels"].to(self.device)
                outputs        = self.model(input_features=input_features, labels=labels)
                total_loss    += outputs.loss.item()

                if i % self.print_every == 0:
                    print(f"  [Eval {i}/{len(self.test_loader)}] Avg loss: {total_loss / i:.4f}")

        avg = total_loss / len(self.test_loader)
        print(f"[WhisperLoRATrainer] Evaluation loss: {avg:.4f}")
        return avg

    def save(self, path: Optional[str] = None) -> str:
        """
        Save (overwrite) the LoRA adapter and update the seen-files ledger.

        Parameters
        ----------
        path : str, optional
            Target directory. Defaults to ``{output_dir}/lora_adapter``.

        Returns
        -------
        str
            The directory the adapter was saved to.
        """
        self._require_setup()
        save_path = path or os.path.join(self.output_dir, _ADAPTER_SUBDIR)

        self.model.save_pretrained(save_path)
        print(f"[WhisperLoRATrainer] Adapter saved → {save_path}")

        # Mark this run's files as seen only after a successful save, so a
        # crashed run doesn't silently skip files on the next attempt.
        self._update_seen_files(self._current_run_files)

        return save_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> List[Tuple[str, str]]:
        """Parse the metadata JSON and return (audio_path, transcript) pairs."""
        meta_path = Path(self.meta_file) 
        
        with open(meta_path, "r") as f:
            metadata = [json.loads(line) for line in f if line.strip()]

        data = []
        for item in metadata:
            filename   = item.get("wav_file")
            transcript = item.get("transcript", "")
            if not filename or not transcript:
                continue
            audio_path = os.path.join(self.data_dir, filename)
            if os.path.exists(audio_path):
                data.append((audio_path, transcript))
            else:
                print(f"  [WARN] Missing file, skipping: {audio_path}")

        return data

    def _split(self, data: List[Tuple[str, str]]) -> Tuple[list, list]:
        """Shuffle and split data into train/test."""
        random.seed(self.seed)
        shuffled = data.copy()
        random.shuffle(shuffled)
        split = int(len(shuffled) * (1 - self.test_split))
        return shuffled[:split], shuffled[split:]

    def _build_model(self) -> PeftModel:
        """
        Load Whisper-small and attach LoRA adapters.

        If a previously saved adapter exists in output_dir it is loaded so
        training continues from the last run rather than from the base model.
        """
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        base.to(self.device)

        adapter_path = os.path.join(self.output_dir, _ADAPTER_SUBDIR)

        if os.path.isdir(adapter_path):
            # Resume from the adapter saved by the previous run
            print(f"[WhisperLoRATrainer] Resuming from existing adapter: {adapter_path}")
            model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        else:
            # First run — initialise a fresh adapter
            print("[WhisperLoRATrainer] No existing adapter found — initialising fresh LoRA weights.")

            # LoRA adapters are applied only to the decoder's attention projections.
            lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
            model = get_peft_model(base, lora_config)

        model.print_trainable_parameters()
        return model

    def _sample_replay_data(self, old_data: List[Tuple[str, str]],
                             n_new: int) -> List[Tuple[str, str]]:
        """
        Draw a random subset of previously-seen samples for experience replay.

        The buffer size is ``replay_ratio * n_new``, rounded down, and capped
        at the total number of available old samples. Returns an empty list on
        the first run (no old data yet) or when replay_ratio is 0.
        """
        if not old_data or self.replay_ratio <= 0.0:
            return []
        n_replay = min(int(n_new * self.replay_ratio), len(old_data))
        return random.sample(old_data, n_replay)

    def _load_seen_files(self) -> Set[str]:
        """Return the set of filenames already trained on, from the ledger."""
        ledger_path = os.path.join(self.output_dir, _SEEN_FILES_LEDGER)
        if not os.path.exists(ledger_path):
            return set()
        with open(ledger_path, "r") as f:
            return set(json.load(f))

    def _update_seen_files(self, new_filenames: List[str]) -> None:
        """Append new_filenames to the seen-files ledger."""
        seen = self._load_seen_files()
        seen.update(new_filenames)
        ledger_path = os.path.join(self.output_dir, _SEEN_FILES_LEDGER)
        with open(ledger_path, "w") as f:
            json.dump(sorted(seen), f, indent=2)
        print(f"[WhisperLoRATrainer] Ledger updated — {len(seen)} total files seen → {ledger_path}")

    def _train_epoch(self, epoch: int) -> float:
        """Run one epoch and return the average loss."""
        self.model.train()
        running_loss = 0.0
        total_loss   = 0.0

        for i, batch in enumerate(self.train_loader, start=1):
            self._global_step += 1

            input_features = batch["input_features"].to(self.device)
            labels         = batch["labels"].to(self.device)

            outputs = self.model(input_features=input_features, labels=labels)
            loss    = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_loss   += loss.item()

            if i % self.print_every == 0:
                avg = running_loss / self.print_every
                print(f"  [Epoch {epoch} | Batch {i}/{len(self.train_loader)}] Avg loss: {avg:.4f}")
                running_loss = 0.0

            if self._global_step % self.checkpoint_every == 0:
                ckpt = os.path.join(self.output_dir, f"checkpoint_step{self._global_step}.pt")
                torch.save(self.model.state_dict(), ckpt)
                print(f"  Checkpoint saved → {ckpt}")

            if self.on_batch_end:
                self.on_batch_end(epoch, i, self._global_step, loss.item())

        return total_loss / len(self.train_loader)

    def _require_setup(self) -> None:
        if self.model is None:
            raise RuntimeError("Call setup() before train() / evaluate() / save().")

# TODO: Remove entry point once integrated with main application
# ---------------------------------------------------------------------------
# Entry point — run directly to test: python lora_trainer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Optional: simple callbacks so you can see per-batch and per-epoch hooks fire
    def on_batch(epoch, batch_idx, step, loss):
        pass  # already printed inside the trainer; add custom logging here if needed

    def on_epoch(epoch, avg_loss):
        print(f"  → Epoch {epoch} callback received avg_loss={avg_loss:.4f}")

    trainer = WhisperLoRATrainer(
        data_dir     = "../../data/training/audio",
        meta_file    = "../../data/training/samples.jsonl",
        output_dir   = "../../data/training/checkpoints",
        on_epoch_end = on_epoch,
    )

    trainer.setup()

    if trainer.has_new_data:
        print("\n--- Training ---")
        losses = trainer.train()
        print(f"Per-epoch losses: {[round(l, 4) for l in losses]}")

        print("\n--- Evaluating ---")
        eval_loss = trainer.evaluate()
        print(f"Eval loss: {eval_loss:.4f}")

        print("\n--- Saving ---")
        saved_to = trainer.save()
        print(f"Adapter saved to: {saved_to}")
    else:
        print("No new data found — nothing to train on.")
        print("Add new entries to your metadata.json and re-run to test incremental training.")

