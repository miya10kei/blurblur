# blurblur

駐車場画像のナンバープレート・人物を自動検出し、モザイク処理するCLIツールです。

YOLOv8のONNXモデルを使用して対象を検出し、検出領域にピクセル化によるモザイク効果を適用します。

## 必要要件

- Python 3.11以上
- [uv](https://github.com/astral-sh/uv)
- [mise](https://mise.jdx.dev/)（タスクランナー）

## セットアップ

```bash
mise run setup
```

以下が実行されます。

1. 依存パッケージのインストール（`uv sync`）
2. YOLOモデルのダウンロード

ダウンロードされるモデル:

| モデル | 用途 |
|--------|------|
| `yolov8n.onnx` | 人物検出（YOLOv8 nano） |
| `plate_detect.onnx` | ナンバープレート検出 |

## 使い方

```bash
uv run blurblur \
  --input-dir ./test_images \
  --output-dir ./test_output \
  --person-model ./models/yolov8n.onnx \
  --plate-model ./models/plate_detect.onnx
```

### オプション

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--input-dir` | （必須） | 入力画像ディレクトリ |
| `--output-dir` | （必須） | 出力画像ディレクトリ |
| `--person-model` | （必須） | 人物検出ONNXモデルのパス |
| `--plate-model` | （必須） | ナンバープレート検出ONNXモデルのパス |
| `--conf-threshold` | `0.25` | 検出の信頼度閾値（0.0〜1.0） |
| `--iou-threshold` | `0.45` | NMSのIoU閾値（0.0〜1.0） |
| `--mosaic-ratio` | `0.05` | モザイクの強度。小さいほど粗い（0.0〜1.0） |

### 対応画像形式

`.jpg` `.jpeg` `.png` `.bmp` `.webp`

## 開発

### miseタスク

| タスク | 説明 |
|--------|------|
| `mise run setup` | 依存インストール + モデルDL |
| `mise run test` | テスト実行 |
| `mise run lint` | Lintチェック |
| `mise run lint:fix` | Lint自動修正 |
| `mise run format` | コードフォーマット |
| `mise run format:check` | フォーマットチェック |
| `mise run typecheck` | 型チェック |
| `mise run check` | 全チェック一括実行 |

### プロジェクト構成

```
blurblur/
├── pyproject.toml
├── mise.toml
├── src/                   # ソースコード（blurblurパッケージ）
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py             # CLIエントリーポイント
│   ├── detection.py       # 検出結果データモデル
│   ├── detector.py        # YOLOv8検出器
│   └── mosaic.py          # モザイク処理
├── scripts/
│   └── download_models.py # モデルDLスクリプト
├── models/                # ONNXモデル
├── tests/                 # テスト
└── test_images/           # テスト用画像
```
