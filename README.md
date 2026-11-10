[![DOI](https://zenodo.org/badge/1020293812.svg)](https://doi.org/10.5281/zenodo.24119777)

# nuclei-segmenter
Pipeline for segmenting nuclei and masking corresponding fluorescent signals

# Nuclei Segmentation Pipeline

This pipeline processes multi-channel 3D TIFF images, performing segmentation, masking, and reporting for nuclei and associated biomarkers. Follow these steps to set up and run the pipeline on your data.

## Prerequisites

* **Environment:** Ensure you have `conda` or `mamba` installed.
* **GPU:** A CUDA-capable GPU is required for prediction.
* **Data Preparation:** Create a copy of your TIFF files in a dedicated working directory to avoid altering source data.

## Setup

Activate the conda environment:

```bash
mamba activate your-env-name
```

Replace `your-env-name` with the name of your conda/mamba environment containing necessary dependencies (napari, PyTorch, etc.).

Navigate to the directory containing the segmentation scripts:

```bash
cd /path/to/nuclei-segmenter
```

Replace `/path/to/nuclei-segmenter` with the actual path where you've cloned this repository.

## Processing Steps

### 0. Download Model Weights

The model weights are too large to host on GitHub so please download the `root.tar` file from this [link](https://drive.google.com/file/d/1cKxFwN0p7n7Qn1Ix3SWRCGPISG3qvWjH/view?usp=sharing).    

Then untar the file called root somewhere convenient. You will need them later during the prediction step.

### 1. Split TIFF Channels

Break multi-channel TIFF stacks into individual channels:

```bash
python split_tiff_channels.py --input_dir /path/to/your/working/dir
```

Replace `/path/to/your/working/dir` with your actual working directory path.

This script creates individual subdirectories for each TIFF, containing separated channel stacks. Once completed, remove original multi-channel TIFF files:

```bash
rm /path/to/your/working/dir/*.tif
```

### 2. Predict

Run predictions using a pretrained model:

```bash
python predict.py --device 0 --input_dir /path/to/your/working/dir --channel_id 1 --root_dir /path/to/model/weights/root
```

Parameters:

* `device`: GPU number (e.g., 0)
* `input_dir`: Path to your working directory
* `channel_id`: Channel number for nuclei segmentation
* `root_dir`: Where you put the model weights `root` directory from Step 0

Ensure the pretrained model file is available or adjust the path accordingly in `predict.py`.

### 3. Post-Processing

Post-process predictions to form cohesive 3D nuclei blobs and remove small artifacts:

```bash
python post_process.py --input_dir /path/to/your/working/dir
```

### 4. Masking and Reporting

Calculate average and summed intensities for nuclei regions within your specified biomarker channel:

```bash
python mask_and_report.py --input_dir /path/to/your/working/dir --background_value 101 --channel_id 2
```

Parameters:

* `input_dir`: Path to your working directory
* `background_value`: Pixel intensity value representing background (adjust as needed)
* `channel_id`: Channel number of biomarker for masking and intensity reporting

A `mask_report.csv` will be created within each image subdirectory.

### 5. Aggregate Results

Combine individual reports into a summarized CSV for the entire experiment:

```bash
python aggregate.py --input_dir /path/to/your/working/dir
```

This produces:

* `condition_summary.csv`: Breakdown by position and experimental condition
* `total_masked_results.csv`: Comprehensive data across all images
