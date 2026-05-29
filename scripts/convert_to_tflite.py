"""Convert a PyTorch model to TFLite via ONNX.

    PyTorch  ──onnx──►  ONNX  ──tf─►  TF SavedModel  ──tflite─►  TFLite

Optional INT8 quantization with a representative dataset (the script
auto-generates random calibration data of the right shape if none is
supplied — replace with real-domain samples for production).

Usage:
    python -m scripts.convert_to_tflite --model vision_emotion --quantize int8
    python -m scripts.convert_to_tflite --model audio_emotion  --quantize fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from backend.models.audio_emotion import AudioEmotionCNN, AudioEmotionModel
from backend.models.vision_emotion import VisionEmotionCNN, VisionEmotionModel


def build(model_name: str) -> Tuple[torch.nn.Module, torch.Tensor]:
    if model_name == "vision_emotion":
        m = VisionEmotionCNN()
        dummy = torch.zeros(1, 1, VisionEmotionModel.INPUT_SIZE, VisionEmotionModel.INPUT_SIZE)
    elif model_name == "audio_emotion":
        m = AudioEmotionCNN()
        dummy = torch.zeros(1, 1, AudioEmotionModel.INPUT_MELS, AudioEmotionModel.INPUT_FRAMES)
    else:
        raise SystemExit(f"unknown model {model_name}")
    return m, dummy


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["vision_emotion", "audio_emotion"])
    p.add_argument("--weights", default=None, help="optional .pt weights to load before export")
    p.add_argument("--quantize", choices=["none", "fp16", "int8"], default="none")
    p.add_argument("--onnx-out", default=None)
    p.add_argument("--tflite-out", default=None)
    args = p.parse_args()

    m, dummy = build(args.model)
    if args.weights:
        m.load_state_dict(torch.load(args.weights, map_location="cpu"))
    m.eval()

    onnx_path = Path(args.onnx_out or f"backend/models/weights/{args.model}.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"wrote ONNX → {onnx_path}")

    # ONNX -> TF -> TFLite
    try:
        import onnx
        from onnx_tf.backend import prepare  # type: ignore
        import tensorflow as tf  # type: ignore
    except ImportError:
        print(
            "Optional deps missing. To enable TFLite export, run:\n"
            "  pip install onnx onnx-tf tensorflow"
        )
        return 0

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_dir = onnx_path.with_suffix("").as_posix() + "_tf"
    tf_rep.export_graph(tf_dir)
    print(f"wrote TF SavedModel → {tf_dir}")

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    if args.quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        def representative():
            for _ in range(100):
                yield [np.random.randn(*dummy.shape).astype(np.float32)]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    tflite_path = Path(args.tflite_out or f"backend/models/weights/{args.model}.tflite")
    tflite_path.write_bytes(tflite_bytes)
    print(f"wrote TFLite ({args.quantize}) → {tflite_path}  ({len(tflite_bytes) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
