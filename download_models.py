"""ONNXモデルをHugging Faceからダウンロードするスクリプト."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = {
    "yolov8n.onnx": "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx",
    "plate_detect.onnx": "https://huggingface.co/ml-debi/yolov8-license-plate-detection/resolve/main/best.onnx",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  スキップ（既に存在）: {dest}")
        return
    print(f"  ダウンロード中: {url}")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  完了: {dest} ({size_mb:.1f} MB)")


def main() -> None:
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    print(f"モデル保存先: {models_dir}")

    for filename, url in MODELS.items():
        dest = models_dir / filename
        try:
            download(url, dest)
        except Exception as e:
            print(f"  エラー: {filename} のダウンロードに失敗 - {e}", file=sys.stderr)
            sys.exit(1)

    print("\n全モデルのダウンロードが完了しました。")


if __name__ == "__main__":
    main()
