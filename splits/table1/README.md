# Table 1 split manifests

These files record the exact image-to-partition assignments for `cancer_vs_non_cancer`. Local image paths are omitted; image IDs, lesion IDs, and labels are retained.

| Partition | Images | Unique `lesion_id` | Positive images |
| --- | ---: | ---: | ---: |
| train | 7,771 | 5,793 | 1314 |
| val | 964 | 724 | 166 |
| test | 953 | 725 | 147 |

Target definition: Cancer (melanoma or basal cell carcinoma) versus non-cancer (nevus, benign keratosis, dermatofibroma, or vascular lesion). The mixed akiec category is excluded because it combines actinic keratoses with intraepithelial carcinoma.

The split seed is `42`. All three pairwise `lesion_id` intersections are zero. `manifest.json` records counts and SHA-256 digests.
