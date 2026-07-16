# ABCDE-Tree

ABCDE-Tree is a local research codebase for testing whether medical vision embeddings improve an interpretable melanoma decision tree beyond dermatologist-inspired ABCDE image features alone.

The main pipeline is:

```text
HAM10000 image
  -> ABCDE feature extractor
  -> medical/foundation-model image embedding extractor
  -> feature fusion
  -> shallow interpretable decision tree and ML baselines
```

This project is for research purposes only. It is not approved for clinical diagnosis.

Clinical deployment would require external validation, dermatologist comparison, bias testing across skin tones and patient populations, prospective validation, clinical workflow testing, and regulatory review.

Black-box models may face greater transparency, documentation, human-oversight, and risk-management burdens in high-risk clinical AI settings. Interpretable models may make clinician oversight and model review easier.

## Scientific Motivation

The main experimental question is:

Does adding medical foundation-model embeddings improve an interpretable decision tree beyond handcrafted ABCDE features alone, while keeping the final classifier interpretable?

The code trains the same tabular classifiers on the same cached representation so model comparisons are fair:

- ABCDE only
- medical embeddings only
- ABCDE + medical embeddings
- ABCDE + medical embeddings + metadata
- raw25_pixels (each image is converted to grayscale, normalized, and then passed into classical ML models)
- pixels ($64 \times 64$ raw images that are passed into neural models like RNN/LSTM/ResNet18 etc.)
 
The main pipeline also trains neural sequence baselines by default on the same prepared features and raw-image
neural baselines on pixels, so the final leaderboard compares shallow ML models, RNN, LSTM, Transformer-style
models, ViT, and CNN with the same metrics.

The primary proposed model is a shallow `DecisionTreeClassifier` with `class_weight="balanced"`.

## Environment

This guide has been tested for **Ubuntu or WSL2 (Ubuntu)** with **Python 3.10**. An NVIDIA GPU is recommended but not required; CPU runs work but can take much longer. The first run needs an internet connection to download the image-embedding model.

### 1. Get the code

If you have not already downloaded the project, open a terminal and run:

```bash
git clone https://github.com/biswasshreyan0/skin-cancer-paper.git
cd skin-cancer-paper
```

If you already have the project, open a terminal in the folder containing `README.md`. All commands below must be run from that folder.

### 2. Create the Python environment

Target platform:

- WSL Ubuntu
- Conda
- Python 3.10
- NVIDIA GPU with 8 GB VRAM recommended for the full pipeline

Create and activate the environment:

```bash
conda create -n skinai python=3.10 -y
conda activate skinai
```

Install PyTorch with CUDA support:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Verify that PyTorch can see the GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
```

An RTX 4060 with 8 GB VRAM should be enough for embedding extraction with `--batch_size 8`. If CUDA runs out of memory, use `--batch_size 4`.

If CUDA is not available, you can still run the project on CPU by adding `--device cpu --batch_size 2` to the commands below. Expect it to be substantially slower.

## Dataset

Follow the beginner-friendly download instructions in [DOWNLOAD_DATA.md](DOWNLOAD_DATA.md). When complete, your files should look like this:

```text
data/HAM10000/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
    HAM10000_images_part_2/
```

The loader creates:

- `binary_label = 1` for `dx == "mel"`
- `binary_label = 0` otherwise
- `multiclass_label = dx`

Processed metadata is saved to:

```text
data/processed/ham10000_metadata_processed.csv
```

Splits are made by `lesion_id`, not `image_id`, so multiple images of the same lesion cannot leak across train, validation, and test.

## Run

### First run: fast sanity check

This first command checks that the dataset, feature extraction, embeddings, and classical models work. It skips the slower neural models:

```bash
python scripts/train_all.py \
    --data_root data/HAM10000 \
    --out_dir outputs \
    --batch_size 8 \
    --pca_dim 64 \
    --tabular_models decision_tree \
    --skip_sequence_models \
    --skip_raw_image_models \
    --skip_cnn
```

On success, you will see files such as `outputs/reports/leaderboard.csv` and `outputs/figures/decision_tree.png`.

### Full experiment

This trains the classical and neural models. It may take hours and requires considerably more GPU memory, disk space, and time than the sanity check:

```bash
python scripts/train_all.py \
    --dataset ham10000 \
    --data_root data/HAM10000 \
    --embedding_model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --out_dir outputs \
    --task binary \
    --batch_size 8 \
    --pca_dim 64
```

By default this trains:

- Classical tabular baselines on all prepared feature sets, including `raw25_pixels`
- Feature-sequence RNN, LSTM, and Transformer baselines on `abcde_embeddings`
- Raw-image RNN, LSTM, Transformer, and ViT baselines
- CNN ResNet18 baseline

To train the sequence baselines on additional prepared feature sets:

```bash
python scripts/train_all.py \
    --dataset ham10000 \
    --data_root data/HAM10000 \
    --embedding_model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --out_dir outputs \
    --task binary \
    --batch_size 8 \
    --pca_dim 64 \
    --sequence_feature_sets abcde embeddings abcde_embeddings abcde_pca_embeddings abcde_embeddings_metadata
```

Use `--skip_sequence_models` for a faster run with only the classical tabular baselines.
Use `--skip_raw_image_models` to skip raw-image RNN/LSTM/Transformer/ViT, and `--skip_cnn` to skip CNN.

Use `--force_recompute_embeddings` to refresh cached embeddings and `--force_recompute_abcde` to refresh cached ABCDE features.

## Models

Tabular models:

- Decision tree
- Logistic regression
- KNN
- Random forest
- XGBoost, when installed
- LightGBM, when installed
- CatBoost, when installed

Neural feature-sequence baselines:

- RNN
- LSTM
- Transformer encoder

Raw image neural baselines:

- Raw-image RNN
- Raw-image LSTM
- Raw-image Transformer
- Compact ViT
- Torchvision ResNet18 with a frozen backbone and trained classifier head

Supported embedding model names include:

- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- `openai/clip-vit-base-patch32`
- `medclip:default` or `medclip-vit:default`, when the optional `medclip` package is installed

The embedding loader uses OpenCLIP for BiomedCLIP-style models and Hugging Face `AutoProcessor`/`AutoModel`
for standard vision models. Some Hub repositories, including `yyupenn/whylesionclip`, may not expose weights in
a format this loader can consume directly.

## Outputs

The pipeline writes:

```text
outputs/
    embeddings/      cached foundation-model embeddings
    features/        cached ABCDE features
    models/          trained models and fitted feature transforms
    reports/         metrics, leaderboard, rules, coefficients, importances
    splits/          train.csv, val.csv, test.csv
    predictions/     per-model prediction CSVs
    figures/         confusion matrices, ROC curves, tree plots
```

Key report files:

- `outputs/reports/leaderboard.csv`
- `outputs/reports/metrics.json`
- `outputs/reports/tree_rules.txt`
- `outputs/figures/decision_tree.png`
- `outputs/reports/logistic_coefficients.csv`
- `outputs/reports/feature_importances.csv`

Print a compact leaderboard:

```bash
python scripts/make_report.py --out_dir outputs
```
