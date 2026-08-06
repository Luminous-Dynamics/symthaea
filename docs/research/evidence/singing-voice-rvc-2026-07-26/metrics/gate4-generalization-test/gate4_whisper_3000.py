from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="int8")
files = {
    "vocoder": "/var/lib/symthaea/training-runs/diffsinger/gate4_out/step3000_vocoder.wav",
    "griffinlim": "/var/lib/symthaea/training-runs/diffsinger/gate4_out/step3000_griffinlim.wav",
}
for name, path in files.items():
    segs, info = model.transcribe(path, language="en", beam_size=5)
    text = " ".join(s.text.strip() for s in segs)
    print(f"{name}: {text!r}")
