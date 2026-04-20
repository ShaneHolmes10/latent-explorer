# latent-explorer
An unsupervised tool for learning and visualizing the latent structure of image datasets. Train a decoder to reconstruct images from a compressed representation, apply PCA to discover the most meaningful axes of variation, and interactively explore what the model learned.

## Project Structure
 
```
latent-explorer/
    data/
        faces/
            raw/            # Original unprocessed images
            processed/      # Resized/cropped images ready for training
    output/
        faces/
            checkpoints/    # Resumable training checkpoints
            backups/        # Manual backups
            runs/           # Completed training runs
                2026_04_18_1430/
                    model.pt
                    meta.yaml
    src/
        models/
            decoder.py      # Default decoder architecture
        utils/
            data_loader.py  # Dataset loading and preprocessing
            model_utils.py  # Dynamic model loading, checkpoints, run management
            plotting.py     # Training curve visualization
        config.py           # Default hyperparameters
        train.py            # Training entry point
        play.py             # Interactive PCA exploration GUI
    README.md
    requirements.txt
```
 
## Usage
 
### Training
 
Run from the `src/` directory:
 
```
python train.py --dataset faces --model decoder --epochs 200 --latent_dim 80
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
 
### Interactive Exploration
 
```
python play.py --run output/faces/runs/2026_04_18_1430 --model decoder --components 20
```
 
Available flags:
 
| Flag | Default | Description |
|------|---------|-------------|
| --run | None | Path to the run folder to load |
| --model | decoder | Model architecture (must match trained model) |
| --dataset | faces | Which dataset |
| --components | 20 | Number of PCA sliders to display |
| --latent_dim | 80 | Latent space size (must match trained model) |
| --image_size | 128 | Image resolution (must match trained model) |
 
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