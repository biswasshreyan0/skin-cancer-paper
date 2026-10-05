# ABCD-Tree

ABCD-Tree evaluates whether computational single-image ABCD features, medical image embeddings, and clinical metadata improve cancer-versus-non-cancer classification on HAM10000. It is research software, not a diagnostic device.

## Exact prediction target

The default task is `cancer_vs_non_cancer`:

- Positive (`cancer_label=1`): melanoma (`mel`) or basal cell carcinoma (`bcc`).
- Negative (`cancer_label=0`): melanocytic nevus (`nv`), benign keratosis (`bkl`), dermatofibroma (`df`), or vascular lesion (`vasc`).
- Excluded: `akiec`, because the [HAM10000 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/) combines actinic keratoses with intraepithelial carcinoma in one label. That mixed category cannot be assigned honestly to either side of a strict cancer/non-cancer target.

The optional `--task melanoma` target is melanoma versus all other diagnoses. `--task multiclass` predicts the seven original diagnostic categories. The deprecated `--task binary` spelling maps to the melanoma task so older commands cannot silently change meaning.

Every final Table 1 row repeats the target definition, positive and negative diagnoses, excluded diagnoses, decision threshold, command, Git commit, dirty-worktree status, and held-out prediction path.

## ABCD feature scope

The handcrafted features cover asymmetry, border, color, and a normalized diameter fraction. Evolution (E) is not used because HAM10000 provides no longitudinal observations of the same lesion over time.

The [American Academy of Dermatology](https://www.aad.org/public/diseases/skin-cancer/abcdes-melanoma) describes the clinical D warning sign as usually greater than 6 mm while noting that melanomas can be smaller. HAM10000 has no physical pixel-to-millimeter calibration, so the code does not fabricate a 6 mm feature. `diameter_fraction` is the lesion bounding-box diameter divided by the 224-pixel analysis width; it is an image-space proxy, not a millimeter measurement.

## Evaluation design

The default run uses three layers of leakage control and uncertainty reporting:

1. Images are grouped by `lesion_id`; a lesion can appear in only one partition or fold.
2. A fixed 80/10/10 train/validation/test split provides the untouched final 10% holdout and its saved predictions.
3. The combined 90% development set undergoes three repeats of nested five-fold `StratifiedGroupKFold`. Hyperparameters are selected on an inner grouped split, preprocessing is refit inside each outer fold, and every outer-fold prediction is saved.

Table 1 contains final held-out metrics and repeated-CV mean ± standard deviation for the tabular model/feature-set comparisons. The computationally expensive neural baselines are reported separately in `leaderboard.csv` and are not mixed into Table 1 without repeated-CV uncertainty estimates. `ablation_table.csv` compares the same selected model across each feature set. `statistical_tests.csv` contains paired two-sided Wilcoxon tests across outer folds, including model-to-model and feature-ablation comparisons, with Holm-adjusted p-values. These tests are exploratory and should be interpreted with effect sizes and the limited number of folds in mind.

The final held-out test set is never used by cross-validation or hyperparameter selection.

## Class imbalance

The default `--imbalance_strategy hybrid` uses inverse-frequency class weights for decision trees, logistic regression, random forest, XGBoost, LightGBM, CatBoost, and neural losses. KNN, which has no class-weight interface, receives random oversampling performed on the training partition only. Validation and test distributions are never resampled. Alternatives are `class_weight`, `oversample`, and `none`.

## Image resolution

The former 25×25 flattened-pixel baseline has been removed from the final experiment. Raw RNN/LSTM/Transformer/ViT models now receive 224×224 images, as does ResNet18. BiomedCLIP preprocessing also uses its native 224×224 input.

Flattening 224×224 RGB images for classical tabular models would create 150,528 columns per image, consume substantial memory, and discard spatial structure. The 224×224 comparisons therefore use image-native neural architectures and embeddings rather than a flattened tabular matrix.

## Setup

Tested on Ubuntu/WSL2 with Python 3.10. An NVIDIA GPU is strongly recommended for embeddings and 224×224 neural models.

```bash
conda create -n skinai python=3.10 -y
conda activate skinai
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download HAM10000 using [DOWNLOAD_DATA.md](DOWNLOAD_DATA.md). The expected hierarchy is:

```text
data/HAM10000/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
    HAM10000_images_part_2/
```

## Full reproducible experiment

Commit the code before the paper run so `run_manifest.json` and every Table 1 row identify an immutable revision. Then run:

```bash
python scripts/train_all.py \
    --dataset ham10000 \
    --data_root data/HAM10000 \
    --embedding_model_name microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --out_dir outputs_cancer \
    --shared_split_dir splits/table1 \
    --task cancer \
    --device cuda \
    --batch_size 8 \
    --pca_dim 64 \
    --imbalance_strategy hybrid \
    --raw_image_size 224 \
    --raw_image_batch_size 16 \
    --sequence_feature_sets abcd embeddings metadata abcd_metadata embeddings_metadata abcd_embeddings abcd_pca_embeddings abcd_embeddings_metadata \
    --cv_folds 5 \
    --cv_repeats 3 \
    --cnn_pretrained \
    --require_clean_git
```

This is intentionally expensive: repeated nested CV multiplies every tabular feature/model fit by 15 outer folds plus inner hyperparameter searches. The 224×224 neural models are trained once for the fixed held-out experiment, not in every CV fold. If 8 GB of VRAM is insufficient, reduce `--raw_image_batch_size` to 8 or 4 and `--batch_size` to 4; this changes throughput but not the target or split membership.

The same command is safe to rerun after sleep, power loss, or interruption. Completed fixed-holdout stages are detected from `reports/metrics.json` and skipped automatically. Cross-validation writes `reports/cv_progress.csv` after every completed feature-set/model job and resumes those jobs on the next invocation; at most the job interrupted at the time is repeated. Use `--no_resume` to force a full recomputation.

A quick end-to-end structural check is:

```bash
python scripts/train_all.py \
    --data_root data/HAM10000 \
    --out_dir outputs_smoke \
    --task cancer \
    --tabular_models decision_tree \
    --cv_models decision_tree \
    --cv_folds 2 \
    --cv_repeats 1 \
    --skip_sequence_models \
    --skip_raw_image_models \
    --skip_cnn
```

## Outputs and provenance

```text
outputs_cancer/
    embeddings/                      cached foundation-model embeddings
    features/                        cached single-image ABCD features
    models/                          trained models and fitted transforms
    predictions/                     final held-out predictions
        cross_validation/            outer-fold predictions for every CV run
    splits/                          exact full split files used in the run
    reports/
        table1_final_results.csv     final Table 1 with target and provenance per row
        table1_provenance.json       Table 1 command, commit, and held-out split hash
        metrics.json                 aggregate and per-class metrics
        leaderboard.csv              compact final-holdout leaderboard
        cv_fold_metrics.csv          one record per outer fold/model/feature set
        cv_summary.csv               mean ± SD across repeated outer folds
        cv_protocol.json             exact CV protocol
        ablation_table.csv           fixed-model feature-set ablation
        statistical_tests.csv        paired Wilcoxon comparisons
        clinical_implications.md     result-aware clinical interpretation
        run_manifest.json            command, commit, hardware, software, split hashes
```

Each validation and test metric record includes per-class precision, recall, F1, and one-vs-rest AUPRC. Each final prediction CSV includes `image_id`, `lesion_id`, original `dx`, exact target, true label, predicted label, and positive-class probability.

Repository-safe split manifests are written to [`splits/table1/`](splits/table1/). They omit machine-specific image paths and include SHA-256 digests plus lesion-overlap counts.

## Computational cost and hardware reporting

Runtime depends heavily on GPU, CPU core count, optional boosting libraries, and whether embeddings are cached. The first run must download and execute BiomedCLIP; later runs reuse its embeddings. Nested tabular CV is mostly CPU-bound. The raw 224×224 models, ResNet18, and embedding extraction are GPU-bound. Running all optional boosters and all sequence feature sets can take many hours and may take more than a day on a laptop CPU.

Rather than publishing a hardware-independent time estimate, the pipeline measures wall-clock training time for every final model and CV fold. `run_manifest.json` records CPU count, RAM, GPU name, CUDA version, Python version, and key package versions. Table 1 includes per-model training seconds; `cv_summary.csv` reports CV training-time mean and standard deviation. Peak GPU memory is not currently measured, so it must not be inferred from these files.

## Clinical interpretation limits

False negatives can delay biopsy or treatment; false positives can cause anxiety, referral, and unnecessary biopsy. The default 0.5 threshold is a reproducible research threshold, not a clinically optimized operating point. Any proposed deployment requires external and prospective validation, calibration and threshold selection for the intended workflow, dermatologist comparison, subgroup analysis across skin tones and populations, out-of-distribution testing, human-factors evaluation, and regulatory review.

HAM10000 is a selected dermoscopy dataset, not a population-screening cohort. Results do not establish safety on smartphone photographs, primary-care populations, rare diagnoses, or lesions outside its seven categories.

## License

Source code is Apache-2.0. HAM10000, pretrained weights, and third-party packages retain their own licenses and terms; see [NOTICE](NOTICE).
