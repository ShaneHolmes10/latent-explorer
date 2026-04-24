# latent-explorer
An unsupervised tool for learning and visualizing the latent structure of image datasets. Train a decoder to reconstruct images from a compressed representation, apply PCA to discover the most meaningful axes of variation, and interactively explore what the model learned.

## Setup
 
### Environment
 
```sh
conda create -n latent-explorer python=3.11
conda activate latent-explorer
pip install -r requirements.txt
```
 
### Faces Dataset
 
Download CelebA from Kaggle into the data directory:
 
```sh
pip install kaggle
kaggle datasets download -d jessicali9530/celeba-dataset -p data/faces/raw/ --unzip
```
 
This places ~200k aligned face images and attribute CSVs into `data/faces/raw/`.
 
Pack the raw images into an HDF5 file:
 
```sh
python data/pack_hdf5.py --input data/faces/raw/img_align_celeba --output data/faces/raw/raw.h5
```
 
Optionally zip the loose images for browsing, then delete them:
 
```sh
cd data/faces/raw
zip -r ../raw_images.zip img_align_celeba/
rm -rf img_align_celeba
```
 
### Preprocessing
 
```sh
python data/faces/preprocessor.py --image_size 128
```
 
This reads from `data/faces/raw/raw.h5`, center crops each image to square, resizes to the target resolution, and writes to `data/faces/processed/processed.h5`.
 
### Viewing Images
 
To verify the data at any stage:
 
```sh
python data/view_h5.py --file data/faces/raw/raw.h5 --top 4
python data/view_h5.py --file data/faces/processed/processed.h5 --random 10
```

## Project Structure
 
```
latent-explorer/
    data/
        base_preprocessor.py        # Base preprocessor class
        pack_hdf5.py                # Convert loose images to HDF5
        view_h5.py                  # View images from HDF5 files
        faces/
            preprocessor.py         # Faces specific preprocessing
            raw/
                raw.h5              # Raw images packed as HDF5
            processed/
                processed.h5        # Preprocessed images as HDF5
    output/
        faces/
            checkpoints/            # Resumable training checkpoints
            backups/                # Manual backups
            runs/                   # Completed training runs
                2026_04_18_1430/
                    model.pt
                    meta.yaml
    src/
        models/
            decoder.py              # Default decoder architecture
            pca_decoder.py          # Decoder with PCA
        utils/
            build_pca_decoder.py    # Constructs PCA decoder model from decoder weights 
            data_loader.py          # Dataset loading
            model_utils.py          # Dynamic model loading, checkpoints, run management
            plotting.py             # Training curve visualization
        config.py                   # Default hyperparameters
        train.py                    # Training entry point
        evaluate.py                 # Evaluate the models reconstruction performance
        play.py                     # Interactive PCA exploration GUI
    pyproject.toml
    README.md
    requirements.txt
    setup_dirs.py
```
 
## Usage
 
### Training
 
```sh
python src/train.py --dataset faces --model decoder --epochs 200 --latent_dim 80
```
 
Available flags:
 
| Flag | Default | Description |
|------|---------|-------------|
| --dataset | faces | Which dataset to use |
| --model | decoder | Which model architecture |
| --epochs | 200 | Number of training epochs |
| --lr | 0.001 | Learning rate |
| --batch_size | 64 | Batch size |
| --latent_dim | 80 | Size of the latent space |
| --image_size | 128 | Image resolution (square) |
| --save_every | 25 | Checkpoint every N epochs |
| --resume | None | Path to checkpoint to resume from |

### Evaluation

Evaluate reconstruction quality by comparing original images against their learned reconstructions.

```sh
# Evaluate 5 random images with a side-by-side plot
python src/evaluate.py --trained output/faces/runs/2026_04_18_1430/model.pt --model decoder --random 5

# Evaluate specific images by index
python src/evaluate.py --trained output/faces/runs/2026_04_18_1430/model.pt --model decoder --indices 0 5000 100000

# Compute MSE and PSNR across the entire dataset
python src/evaluate.py --trained output/faces/runs/2026_04_18_1430/model.pt --model decoder --eval-all

# Combine sample plots with full dataset metrics, and save results
python src/evaluate.py --trained output/faces/runs/2026_04_18_1430/model.pt --model decoder --random 5 --eval-all --save plots/my_eval/
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--trained` | *(required)* | Path to a `.pt` file containing model weights and latent vectors |
| `--model` | `decoder` | Model architecture (must match trained model) |
| `--dataset` | `faces` | Which dataset to load originals from |
| `--latent_dim` | `80` | Latent space size (must match trained model) |
| `--image_size` | `128` | Image resolution (must match trained model) |
| `--random` | `None` | Evaluate N randomly selected images |
| `--indices` | `None` | Evaluate specific images by index (mutually exclusive with `--random`) |
| `--eval-all` | `False` | Compute MSE and PSNR across the entire dataset |
| `--batch_size` | `128` | Batch size for full dataset evaluation |
| `--save` | `None` | Directory to save `reconstruction.png` and `metrics.yaml` |

Outputs per-image MSE and PSNR to the console, a side-by-side reconstruction plot, and optionally a `metrics.yaml` summary file.
 

### Interactive Exploration

```sh
python src/play.py --trained output/faces/runs/2026_04_18_1430/model.pt --model decoder
```

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--trained` | *(required)* | Path to a `.pt` file containing model weights and latent vectors |
| `--model` | `decoder` | Model architecture (e.g. `decoder`, `pca_decoder`) |
| `--dataset` | `faces` | Which dataset |
| `--latent_dim` | `80` | Latent space size (must match trained model) |
| `--image_size` | `128` | Image resolution (must match trained model) |
| `--display_size` | `384` | Size of the displayed image in the GUI |
| `--num_std` | `3.0` | Slider range in standard deviations around the mean |


### PCA Exploration

After training, you can reframe the latent space in terms of principal components — the axes of greatest variation learned by the model. This makes the sliders in the GUI more interpretable: D00 controls the highest-variance dimension, D01 the next, and so on.

#### Step 1 — Build a PCA decoder

```sh
python src/utils/build_pca_decoder.py \
    --input output/faces/runs/2026_04_18_1430/model.pt \
    --output output/faces/runs/2026_04_18_1430/pca_model.pt
```

This fits PCA on the trained latent vectors and saves a new `.pt` file that `play.py` can load directly.

Available flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to a trained decoder `.pt` file |
| `--output` | *(required)* | Path to save the PCA decoder `.pt` file |
| `--latent_dim` | `80` | Latent space size (must match the trained decoder) |
| `--image_size` | `128` | Image resolution (must match the trained decoder) |

#### Step 2 — Launch the GUI with PCA sliders

```sh
python src/play.py \
    --trained output/faces/runs/2026_04_18_1430/pca_model.pt \
    --model pca_decoder \
    --latent_dim 80
```

The sliders now correspond to principal components ordered by explained variance. All other `play.py` flags (e.g. `--num_std`, `--display_size`) work as normal.


## Adding a New Model
 
1. Create a new file in `src/models/` following the naming convention `<model_name>.py`
2. Define a class named `<ModelName>` (PascalCase of the filename)
3. The class must accept `latent_dim` and `image_size` in `__init__`
4. The class must implement `forward(z)` returning an image tensor
5. Use it with `--model <model_name>`
Example: `src/models/vae_decoder.py` containing class `VaeDecoder` is used with `--model vae_decoder`.
 
No registry or config changes needed.
 
## Adding a New Dataset
 
1. Create `data/<dataset_name>/raw/` and place your images there
2. Preprocess into `data/<dataset_name>/processed/`
3. Use it with `--dataset <dataset_name>`
## Run Metadata
 
Each completed run saves a `meta.yaml` with the full training configuration:
 
```yaml
timestamp: 2026-04-18T14:30:00
dataset: faces
model: decoder
latent_dim: 80
image_size: 128
epochs: 200
batch_size: 64
lr: 0.001
num_samples: 202599
final_loss: 0.003421
training_time_seconds: 7234.51
```
 
This ensures reproducibility regardless of filename.


