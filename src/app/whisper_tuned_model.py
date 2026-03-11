import torch
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from typing import Optional

from models.audio_data import AudioStream, Transcript
from utils.logger import get_logger

logger = get_logger("WHISPER_TUNED_MODEL")

class WhisperLoraAsrModel:
    """
    Whisper Small model with LoRA adapters loaded at runtime with local caching to avoid redownloading.
    """

    BASE_MODEL_NAME = "openai/whisper-small"
    CACHE_DIR = "./models/hf_cache" 

    def __init__(
        self,
        model_checkpoint: str,
        device: str = "cpu",
        adapter_path: Optional[str] = None,
        merge_adapters: bool = False,
    ):
        self.device = device

        # Ensure cache directory exists
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.processor = self._load_processor()
        base_model = self._load_base_model()

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

    def _load_processor(self) -> WhisperProcessor:
        """
        Load WhisperProcessor from local cache if available,
        otherwise download and cache it.
        """
        processor_cache_path = os.path.join(self.CACHE_DIR, "processor")
        
        # Check if processor exists locally
        if os.path.exists(processor_cache_path) and os.listdir(processor_cache_path):
            try:
                processor = WhisperProcessor.from_pretrained(processor_cache_path)
                return processor
            except Exception as e:
                print(f"Failed to load cached processor: {e}")
        
        processor = WhisperProcessor.from_pretrained(self.BASE_MODEL_NAME)
        
        # Save processor to cache
        processor.save_pretrained(processor_cache_path)
        
        return processor

    def _load_base_model(self) -> WhisperForConditionalGeneration:
        """
        Load base Whisper model from local cache if available,
        otherwise download and cache it.
        """
        model_cache_path = os.path.join(self.CACHE_DIR, "base_model")
        
        # Check if model exists locally
        if os.path.exists(model_cache_path) and os.listdir(model_cache_path):
            try:
                base_model = WhisperForConditionalGeneration.from_pretrained(model_cache_path)
                return base_model
            except Exception as e:
                print(f"Failed to load cached model: {e}")
        
        base_model = WhisperForConditionalGeneration.from_pretrained(self.BASE_MODEL_NAME)
        
        base_model.save_pretrained(model_cache_path)
        
        return base_model

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

    def decode(self, features, no_speech_threshold: float = 0.6, confidence_threshold: float = 0.2) -> Optional[Transcript]:
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_features=features,
                forced_decoder_ids=forced_decoder_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Use top-1 probability of first token as confidence
        first_token_probs = torch.softmax(outputs.scores[0], dim=-1)
        top_prob = first_token_probs[0].max().item()
        
        logger.debug(f"First token confidence: {top_prob:.4f}")

        if top_prob < confidence_threshold:
            logger.info(f"Low confidence ({top_prob:.4f}), discarding as noise")
            return Transcript(text="",metadata={"confidence": top_prob})

        text = self.processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )[0].strip()

        return Transcript(text=text, metadata={"confidence": top_prob})