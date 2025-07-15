import os
import argparse
import numpy as np
import tifffile
import csv

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Script to mask the selected channel with tracked_masks.tif "
            "and report sum/average per label."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory that contains subdirectories with .tif files.",
    )
    parser.add_argument(
        "--background_value",
        type=int,
        required=True,
        help="An integer background value to subtract from the average intensity only."
    )
    parser.add_argument(
        "--channel_id",
        type=int,
        required=True,
        help="Integer specifying which channel to aggregate values from"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    background_value = args.background_value
    channel_id = int(args.channel_id)

    # Get all subdirectories in the input directory
    subdirs = [
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    for subdir in subdirs:
        masks_path = os.path.join(subdir, "tracked_masks.tif")
        channel_path = os.path.join(subdir, "channel_{}.tif".format(channel_id))

        # Check for existence of required files
        if not (os.path.exists(masks_path) and os.path.exists(channel_path)):
            print(f"Skipping {subdir} because required files (tracked_masks.tif, channel_#.tif) are not found.")
            continue

        print(f"Processing directory: {subdir}")

        # Read the .tif images (assumes 3D stacks, but also works for 2D)
        masks = tifffile.imread(masks_path)
        channel = tifffile.imread(channel_path)

        # Ensure both images have the same shape
        if masks.shape != channel.shape:
            print(f"Warning: Shape mismatch in {subdir}. Skipping.")
            continue

        # Collect the unique labels in tracked_masks (excluding 0)
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels != 0]

        # Prepare CSV output
        output_csv = os.path.join(subdir, "mask_report.csv")
        with open(output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Updated column headers
            writer.writerow(["id", "summed_intensity", "average_intensity"])

            for label in unique_labels:
                # Gather the pixel values from channel corresponding to the current label
                region_values = channel[masks == label]

                # Calculate sum (raw, no background subtraction)
                pixel_sum = np.sum(region_values)
                # Calculate average (raw), then subtract background_value
                pixel_avg = np.mean(region_values) - background_value

                # Convert sum to scientific notation, average to 3 decimals
                pixel_sum_str = f"{pixel_sum:.4e}"          # scientific notation
                pixel_avg_str = f"{pixel_avg:.3f}"          # 3 decimal places

                # Write a row for this label
                writer.writerow([label, pixel_sum_str, pixel_avg_str])

        print(f"Saved report to {output_csv}")


if __name__ == "__main__":
    main()



