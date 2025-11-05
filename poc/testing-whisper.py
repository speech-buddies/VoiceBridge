import whisper
import torch

# print(torch.cuda.is_available())

# Load a pre-trained model (options: "tiny", "base", "small", "medium", "large")
model = whisper.load_model("base")
# "0a71bb2c-5579-468f-3729-08dc13d38337_1077_4303.wav"

# path to your .wav file
audio_path = "2-test.wav"

# transcribe the audio
result = model.transcribe(audio_path)

# print the text
print("Transcription:")
print(result["text"])
