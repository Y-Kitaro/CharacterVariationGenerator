# Character Variation Generator

キャラクターの立ち絵からポーズ差分・表情差分を生成するAIツールです。

## インストール

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/Y-Kitaro/CharacterVariationGenerator.git
   cd CharacterVariationGenerator
   ```

2. **Pytorchのインストール**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
   ```


3. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```

## モデルのセットアップ (手動ダウンロード)

本ツールを使用するには、いくつかのモデルを手動でダウンロードして `assets/models/` ディレクトリ（または指定のパス）に配置する必要があります。

### 1. ディレクトリの作成
```bash
mkdir -p assets/models
```

### 2. Pose Generator (Qwen Image Edit)
*   **モデル**: Qwen-Image-Edit (例: `Qwen/Qwen-Image-Edit-2511`)
*   **手順**: 
    1. HuggingFace等からモデル一式をダウンロードしてください。
    2. `assets/models/qwen_image_edit/` ディレクトリに配置してください。
    ※ 空の `.gitkeep` がありますが、モデルファイル (`config.json`, `*.safetensors` 等) を同じ場所に置いてください。

### 3. Upscaler (Real-ESRGAN)
*   **モデル**: RealESRGAN_x4plus_anime_6B
*   **手順**: 
    以下のリンクから `RealESRGAN_x4plus_anime_6B.pth` をダウンロードし、`assets/models/` に配置してください。
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

### 4. Mask Generator (Meta SAM3)
*   **モデル**: SAM3 (via Ultralytics)
*   **手順**:
    アプリ初回起動時に `ultralytics` ライブラリ経由で自動的に必要なモデルがダウンロードされます。
    もし手動で管理したい場合は、`sam2.pt` (または `sam3.pt`) をダウンロードし、設定ファイルのパスに合わせて配置してください。

### 5. Expression Editor (SDXL)
*   **モデル**: アニメ系SDXLモデル (例: Animagine XL 3.1 など)
*   **手順**:
    Civitaiなどから好みのSDXL Checkpoint (.safetensors) をダウンロードし、任意の場所に保存してください。
    `config/settings.yaml` の `expression_editor.checkpoint_path` にその絶対パスまたは相対パスを記述してください。

## 使い方

```bash
python app.py
```
ブラウザで `http://127.0.0.1:7860` にアクセスしてください。