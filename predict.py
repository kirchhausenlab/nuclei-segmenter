import argparse
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import imageio.v3 as iio
from tifffile import imread, imwrite
import shutil

import torch

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def run_automatic_instance_segmentation(image, checkpoint_path, model_type="vit_b_lm", device=None):
    """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

    NOTE: AIS is supported only for `ï¿^bï¾µsam` models.

    Args:
        image: The input image.
        checkpoint_path: The path to stored checkpoints.
        model_type: The choice of the `ï¿^bï¾µsam` model.
        device: The device to run the model inference.

    Returns:
        The instance segmentation.
    """
    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, # choice of the Segment Anything model
        checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
        device=device,  # the device to run the model inference.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,
        ndim=2,
    )

    return prediction


def main(TOP_DIR, ROOT_DIR, model_type, checkpoint_name, device, best_checkpoint, channel_id):

    jdx = 0

    for tif_stack_dir in os.listdir(TOP_DIR):


        dir_path = os.path.join(TOP_DIR, tif_stack_dir)
        stack = [e for e in os.listdir(dir_path) if e.endswith(".tif")][0]

        tif_stack_path = os.path.join(dir_path, 'channel_{}.tif'.format(channel_id))
        
        print("> > > > >")
        print(tif_stack_path)

        # Make temporary directories
        tmp_img_path = os.path.join(dir_path, "tmp_img")
        tmp_masks_path = os.path.join(dir_path, "tmp_masks")
        os.makedirs(tmp_img_path, exist_ok=True)
        os.makedirs(tmp_masks_path, exist_ok=True)

        # Load TIFF stack and save planes into tmp_img
        tif_stack_data = imread(tif_stack_path)
        for idx, plane in enumerate(tif_stack_data):
            plane_filename = f"{idx:04d}.tif"
            plane_uint8 = plane.astype(np.uint8)  # Convert to uint8
            iio.imwrite(os.path.join(tmp_img_path, plane_filename), plane_uint8)

        # Process each plane in tmp_img and save predictions to tmp_masks
        for plane_filename in sorted(os.listdir(tmp_img_path)):
            image_path = os.path.join(tmp_img_path, plane_filename)
            image = iio.imread(image_path)

            # Run prediction
            prediction = run_automatic_instance_segmentation(
                image=image,
                checkpoint_path=best_checkpoint,
                model_type=model_type,
                device=device,
            )
            p8 = (prediction > 0).astype(np.uint8)

            # Save prediction to tmp_masks
            iio.imwrite(os.path.join(tmp_masks_path, plane_filename), p8)

        # Combine images in tmp_masks into a single z-stack
        mask_planes = [iio.imread(os.path.join(tmp_masks_path, f)) for f in sorted(os.listdir(tmp_masks_path))]
        combined_stack = np.stack(mask_planes, axis=0)

        # Validate size matches the original stack
        assert combined_stack.shape == tif_stack_data.shape, "Error: Combined stack size mismatch!"

        # Save combined stack
        combined_stack_path = os.path.join(dir_path, "combined_masks.tif")
        imwrite(combined_stack_path, combined_stack.astype(np.uint8))

        # Clean up tmp directories
        shutil.rmtree(tmp_img_path)
        shutil.rmtree(tmp_masks_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run a model with configurable device and top directory.")
    parser.add_argument("--device", type=int, required=True, help="CUDA device number (e.g., 1 for cuda:1)")
    parser.add_argument("--input_dir", type=str, required=True, help="Top directory path")
    parser.add_argument("--channel_id", type=int, required=True, help="Channel to segment")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to root directory containing model checkpoint")


    args = parser.parse_args()

    TOP_DIR = args.input_dir
    ROOT_DIR = args.root_dir
    model_type = "vit_h"
    checkpoint_name = "nuclei_ft_feb20"
    device = torch.device(f"cuda:{args.device}")

    best_checkpoint = os.path.join(ROOT_DIR, "models", "checkpoints", checkpoint_name, "best.pt")

    main(TOP_DIR, ROOT_DIR, model_type, checkpoint_name, device, best_checkpoint, int(args.channel_id))
