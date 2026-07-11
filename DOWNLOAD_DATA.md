# Download HAM10000

The pipeline expects this layout:

```text
data/HAM10000/
    HAM10000_metadata.csv
    HAM10000_images_part_1/
    HAM10000_images_part_2/
```

## Important Token Note

If you accidentally pasted a Hugging Face token anywhere public or semi-public, revoke it and create a new one:

```text
Hugging Face -> Settings -> Access Tokens -> delete/revoke the exposed token
```

You do not need a Hugging Face token for the Kaggle download path below.

## Option 1: Kaggle CLI

Install the Kaggle CLI inside the `skinai` environment:

```bash
cd /home/sujoy/abcde-tree
conda activate skinai
pip install kaggle
```

### Kaggle Credentials

Kaggle has two credential styles depending on which settings page/version you see.

#### New Token Screen

If Kaggle shows an API token starting with `KGAT_`, do not paste it into project files. Set it only in your current shell:

```bash
export KAGGLE_API_TOKEN='PASTE_YOUR_KAGGLE_TOKEN_HERE'
```

Then test:

```bash
kaggle competitions list
```

If that works, continue to the download command below.

Some Kaggle CLI versions still error with `Could not find kaggle.json` even after `KAGGLE_API_TOKEN` is exported. In that case, save the token where the client reads it automatically. Put it in both common locations:

```bash
mkdir -p ~/.kaggle ~/.config/kaggle
printf '%s\n' 'PASTE_YOUR_KAGGLE_TOKEN_HERE' > ~/.kaggle/access_token
cp ~/.kaggle/access_token ~/.config/kaggle/access_token
chmod 600 ~/.kaggle/access_token ~/.config/kaggle/access_token
```

Then test again:

```bash
kaggle competitions list
```

#### Older `kaggle.json` Screen

If Kaggle downloads a `kaggle.json` file, use this path instead:

1. Go to `https://www.kaggle.com/settings`
2. Scroll to API
3. Click `Create New Token`
4. This downloads `kaggle.json`

Place `kaggle.json` where the Kaggle CLI expects it:

```bash
mkdir -p ~/.kaggle
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If Kaggle fails with this error:

```text
KeyError: 'username'
```

it means the CLI is installed, but it cannot find your Kaggle API credentials. Confirm the file exists and contains `username` and `key`:

```bash
ls -l ~/.kaggle/kaggle.json
python -m json.tool ~/.kaggle/kaggle.json
```

The JSON should look like:

```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

If your browser downloaded `kaggle.json` to Windows Downloads, replace `YOUR_WINDOWS_USERNAME` and copy it:

```bash
mkdir -p ~/.kaggle
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download and unzip HAM10000:

```bash
cd /home/sujoy/abcde-tree
mkdir -p data/HAM10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
```

Check the files:

```bash
ls data/HAM10000
```

You should see:

```text
HAM10000_metadata.csv
HAM10000_images_part_1
HAM10000_images_part_2
```

## If the ZIP Creates an Extra Folder

If the download creates a nested folder, move the files so they match the expected layout:

```bash
find data/HAM10000 -maxdepth 3 -name HAM10000_metadata.csv
find data/HAM10000 -maxdepth 3 -type d -name HAM10000_images_part_1
find data/HAM10000 -maxdepth 3 -type d -name HAM10000_images_part_2
```

Then move those three items directly under:

```text
/home/sujoy/abcde-tree/data/HAM10000/
```

## Verify With the Project

After the files are in place:

```bash
python scripts/prepare_ham10000.py --data_root data/HAM10000 --out_dir outputs
```

If that succeeds, the full pipeline can find the dataset.

## Option 2: KaggleHub Fallback

If the `kaggle` CLI keeps failing with:

```text
Could not find kaggle.json
```

and Kaggle only gave you a `KGAT_...` token, use KaggleHub instead:

```bash
cd /home/sujoy/abcde-tree
conda activate skinai
pip install kagglehub
python - <<'PY'
import kagglehub

path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print(path)
PY
```

This prints the cache directory where KaggleHub downloaded the dataset. Copy the HAM10000 files into this project:

```bash
mkdir -p data/HAM10000
cp -r PRINTED_PATH/* data/HAM10000/
```

Replace `PRINTED_PATH` with the path printed by the Python command.

Then verify:

```bash
python scripts/prepare_ham10000.py --data_root data/HAM10000 --out_dir outputs
```
