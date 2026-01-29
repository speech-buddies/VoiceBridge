import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from typing import Optional

from src.models.audio_data import AudioStream, Transcript


class WhisperLoraAsrModel:
    """
    Whisper Small model with LoRA adapters loaded at runtime.
    """

    BASE_MODEL_NAME = "openai/whisper-small"

    def __init__(
        self,
        model_checkpoint: str,
        device: str = "cpu",
        adapter_path: Optional[str] = None,
        merge_adapters: bool = False,
    ):
        self.device = device

        self.processor = WhisperProcessor.from_pretrained(self.BASE_MODEL_NAME)
        base_model = WhisperForConditionalGeneration.from_pretrained(self.BASE_MODEL_NAME)

        self.adapter_path = adapter_path
        state_dict = torch.load(self.adapter_path, map_location=self.device)

        vocab_ckpt = state_dict["base_model.model.model.decoder.embed_tokens.weight"].shape[0]
        vocab_base = base_model.config.vocab_size

        if vocab_ckpt != vocab_base:
            base_model.model.decoder.embed_tokens = torch.nn.Embedding(
                vocab_ckpt,
                base_model.config.d_model,
            )
            base_model.model.proj_out = torch.nn.Linear(
                base_model.config.d_model,
                vocab_ckpt,
            )

        # Attach LoRA adapters
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )

        self.model = get_peft_model(base_model, peft_config)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def is_valid(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        # Whisper decoding is stateless
        pass

    def extract_features(self, audio: AudioStream):
        inputs = self.processor(
            audio.samples,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )
        return inputs.input_features.to(self.device)

    def decode(self, features) -> Transcript:
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features=features)

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return Transcript(
            text=text.strip(),
            metadata={"model": "whisper-small-lora"},
        )
