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

## License and Third-Party Materials

The source code is licensed under [Apache-2.0](LICENSE). The HAM10000 dataset, pretrained model weights, and
third-party packages are not included under that license. Download or use them only under their respective licenses,
terms of use, and access requirements. See [NOTICE](NOTICE) for details.

## Scientific Motivation

This project compares several ways of representing and classifying dermatoscopic images from the
[HAM10000 dataset](https://arxiv.org/abs/1803.10417). It supports binary melanoma detection (`mel` versus all
other diagnoses) and seven-class diagnosis prediction. The data is split reproducibly into training, validation,
and test partitions by `lesion_id`, so images of the same lesion cannot appear in more than one partition. Models
are fitted only on training data, tuning and early stopping use validation data, and final metrics are reported on
the held-out test data.

The tabular experiments compare logistic regression, KNN, a shallow class-balanced decision tree, random forest,
and optional XGBoost, LightGBM, and CatBoost models. Each model is trained on the same prepared representations:
downsampled 25×25 image pixels, computational ABCD-style features, foundation-model image embeddings, clinical
metadata, and combinations of those inputs. Imputation, scaling, one-hot encoding, and PCA are fitted on the
training split only and then applied unchanged to validation and test data, avoiding preprocessing leakage.

The project also trains neural baselines: RNN, LSTM, and Transformer encoders over prepared feature sequences;
RNN, LSTM, Transformer, and compact ViT models over raw image pixels; and a ResNet18 CNN baseline. These models
learn their parameters from the training images with validation-based early stopping. All model families write their
predictions and metrics to the same output structure, allowing direct comparison of predictive performance and,
for the shallow decision tree, inspection of its learned decision rules.

The main experimental question is:

> How can we design a novel, reliable method for predicting skin-lesion diagnoses from images?

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

### Complete GPU experiment

To train every model family and evaluate the feature-sequence RNN, LSTM, and Transformer on every prepared feature set, run:

```bash
python scripts/train_all.py \
    --dataset ham10000 \
    --data_root data/HAM10000 \
    --embedding_model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --out_dir outputs_all_gpu \
    --task binary \
    --batch_size 8 \
    --pca_dim 64 \
    --device cuda \
    --sequence_feature_sets raw25_pixels abcde embeddings abcde_embeddings abcde_pca_embeddings metadata abcde_embeddings_metadata
```

This writes the combined comparison to `outputs_all_gpu/reports/leaderboard.csv` and detailed metrics to `outputs_all_gpu/reports/metrics.json`. The neural sequence and raw-image models use CUDA; classical models use their own available backend.

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

## Helpful commands

Prepare the dataset metadata and reproducible train/validation/test splits without training models:

```bash
python scripts/prepare_ham10000.py --data_root data/HAM10000 --out_dir outputs
```

Extract (or reuse) the cached ABCDE features and image embeddings:

```bash
python scripts/extract_features.py --data_root data/HAM10000 --out_dir outputs --device cuda
```

Run decision-tree experiments only:

```bash
python scripts/train_decision_trees.py --data_root data/HAM10000 --out_dir outputs --device cuda
```

Sweep readable decision-tree configurations and save the best rules:

```bash
python scripts/train_readable_tree_sweep.py --data_root data/HAM10000 --out_dir outputs --device cuda
```

Compare logistic regression, KNN, and MLP baselines; add `--run_cnn` to include the CNN:

```bash
python scripts/train_focused_comparison.py --data_root data/HAM10000 --out_dir outputs --device cuda --run_cnn
```

View all options for the main training command:

```bash
python scripts/train_all.py --help
```
