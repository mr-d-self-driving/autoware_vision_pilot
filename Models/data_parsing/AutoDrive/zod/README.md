# ZOD CIPO Radar Pipeline (AutoDrive)

This folder contains scripts to:

1. Associate ZOD camera images with radar + vehicle control timestamps.
2. Run a camera-to-radar CIPO (Closest In-Path Object) association (clustering + temporal tracking).
3. Generate per-image JSON labels for training.
4. Provide optional debug visualizations (camera overlays + radar BEV).

## Required inputs

You need a `--zod-root` directory that contains (at minimum):

- `images_blur_*/sequences/{seq}/camera_front_blur/` (images may be split into 3 ranges)
- `radar_front/sequences/{seq}/radar_front/` (radar `.npy` and timestamps)
- `infos/sequences/{seq}/calibration.json`
- `vehicle_data/sequences/{seq}/vehicle_data.hdf5`
- `associations/{seq}_associations.json` (only required if you skip `step1`)
- `models/autodrive.pt` — AutoSpeed weights for `run_cipo_radar` / debug viz (or pass `--model-path`)

The scripts automatically handle the “3 image segment” layout via `zod_utils.get_images_blur_dir()`.

## Environment / dependencies

Python packages (approximate):

- `numpy`
- `Pillow`
- `opencv-python` (debug viz)
- `scikit-learn` (DBSCAN clustering)
- `h5py` (step1 associations)

## 1) Run the full pipeline for one sequence

```bash
python run_full_pipeline.py \
  --sequence 000490 \
  --zod-root /path/to/zod
```

All generated data is written under **`--zod-root`** (not next to these scripts):

- `{zod_root}/associations/{seq}_associations.json` (step 1)
- `{zod_root}/output/{seq}/cipo_radar.json`
- `{zod_root}/output/{seq}/debug_viz/*.png` (unless `--skip-viz`)
- `{zod_root}/output/{seq}/debug_labels/*.png` (unless `--skip-debug-labels`)
- `{zod_root}/labels/{seq}/*.json` (merged training labels)

Use `--output-base` on `run_full_pipeline.py` to change the parent of per-sequence folders (default: `{zod_root}/output`).

Step 1 (`step1_timestamp_association.py`) lives in this folder; you only need `--zod-root` pointing at the dataset.

## 2) Run only association + CIPO-radar for one sequence

```bash
python run_cipo_radar.py \
  --sequence 000490 \
  --zod-root /path/to/zod
```

Default output: `{zod_root}/output/{seq}/cipo_radar.json`. Pass `--output` to override.

Note: `run_cipo_radar.py` expects `{zod_root}/associations/{seq}_associations.json` from `step1_timestamp_association.py`.

## 3) Debug visualizations

### Camera + CIPO-radar BEV overlay

```bash
python debug_cipo_radar_viz.py \
  --sequence 000490 \
  --zod-root /path/to/zod \
  --every 10
```

Optional:

- `--model-path /path/to/autodrive.pt` (default: `{zod_root}/models/autodrive.pt` — place the checkpoint next to your dataset)

### Label sanity check (no inference)

```bash
python debug_labels_viz.py \
  --sequence 000490 \
  --zod-root /path/to/zod \
  --n 20
```

This reads:

- `{zod_root}/labels/{seq}/*.json`
- `{zod_root}/output/{seq}/cipo_radar.json` (or `--cipo-radar` if provided)

## 4) Run over a sequence range / all sequences

Run a range (inclusive):

```bash
python run_full_pipeline.py \
  --all-sequences \
  --start-seq 000000 \
  --end-seq 000490 \
  --zod-root /path/to/zod \
  --workers 1
```

`zod_utils.iter_all_sequences()` enumerates `000000` to `001472`.

## Notes on image segmentation

Camera images are assumed to be stored under 3 folders based on sequence range:

- `images_blur_000000_000490`
- `images_blur_000491_000981`
- `images_blur_000982_001472`

`zod_utils.get_images_blur_dir(zod_root, seq)` selects the correct segment automatically.
