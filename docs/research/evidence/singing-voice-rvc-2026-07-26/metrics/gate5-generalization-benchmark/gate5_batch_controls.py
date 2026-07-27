#!/usr/bin/env python3
"""Gate 5 controls: render control1/control2 .ds files through the
trained vocoder at given checkpoints (inference only, model loaded once
per checkpoint)."""
import os
import sys
import json

DS = "/var/lib/symthaea/training-runs/diffsinger"
sys.path.insert(0, f"{DS}/DiffSinger")
os.chdir(f"{DS}/DiffSinger")

from utils.hparams import set_hparams, hparams
from inference.ds_acoustic import DiffSingerAcousticInfer
from utils.infer_utils import save_wav

CONTROLS = {
    "control1_seenlyrics_unseenmelody": f"{DS}/gate5_control1_seenlyrics_unseenmelody.ds",
    "control2_unseenlyrics_simplemelody": f"{DS}/gate5_control2_unseenlyrics_simplemelody.ds",
}


def main():
    step = int(sys.argv[1])
    out_dir = f"{DS}/gate5_out"
    os.makedirs(out_dir, exist_ok=True)

    sys.argv = [sys.argv[0], "--exp_name", "benchmark01-gate5", "--infer"]
    set_hparams()

    infer_ins = DiffSingerAcousticInfer(load_vocoder=True, ckpt_steps=step)

    for name, ds_path in CONTROLS.items():
        params = json.load(open(ds_path))
        batch = infer_ins.preprocess_input(params[0], idx=0)
        mel_pred = infer_ins.forward_model(batch)
        wav = infer_ins.run_vocoder(mel_pred, f0=batch["f0"])[0].cpu().numpy()
        save_wav(wav, f"{out_dir}/{name}_step{step}_vocoder.wav", hparams["audio_sample_rate"])
        print(f"[{step}] rendered {name} -> vocoder wav")


if __name__ == "__main__":
    main()
