## AutoDrive 2.0 - temporal end-to-end distance and curvature estimation

For autonomous cruise and driver-assistance applications, it is important to estimate both the road-following demand and the lead-object distance from camera input. AutoDrive is a compact temporal model that predicts three outputs from consecutive front-view frames: **distance-to-CIPO**, **road curvature**, and **CIPO presence probability**.

AutoDrive processes a pair of frames (`t-1`, `t`) and uses a shared-feature temporal fusion head to capture short-term motion cues that a single-frame model can miss. It is designed for 2:1 input aspect ratio and is typically used with the same center-crop preprocessing used during training.

### Demo Video
[Watch the demo video](<ADD_DRIVE_DEMO_LINK_HERE>)

## Get Started

To quickly try AutoDrive on your own data, please follow the steps in the [tutorial](tutorial.ipynb).  
For best results, ensure your inference input follows the training preprocessing pipeline (50-degree center crop and 1024x512 resize).

### Performance Results

AutoDrive is trained as a multi-task regression/classification model on sequence data with labels for curvature, CIPO distance, and CIPO presence.  
Please add your official release metrics here once final evaluation is complete.

## Model variants

AutoDrive currently uses one primary variant in this repository:

**AutoDrive 2.0 model weights - 2:1 aspect ratio, 1024px by 512px input image**
- [Link to Download Pytorch Model Weights *.pt](<https://drive.google.com/drive/u/1/folders/182h_9eBHroMCOfQHJiXVgrNq7zHx7Qws?dmr=1&ec=wgc-drive-hero-goto>)

### Notes

- Training entry point: `Models/training/train_auto_drive.py`
- Core network: `Models/model_components/autodrive/autodrive_network.py`
- Data preprocessing and scaling: `Models/data_utils/load_data_auto_drive.py`
- AutoSpeed backbone warm-start is supported through `--autospeed-ckpt`
