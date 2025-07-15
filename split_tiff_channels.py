import os
import argparse
import shutil
import tifffile

def split_channels_in_directory(input_dir):
    """
    For each .tif stack in `input_dir`:
    - Remove whitespace from the filename (for the base name).
    - Create a subfolder named after the TIFF (whitespace removed, no extension).
    - Copy the .tif file into that subfolder (also with whitespace removed).
    - Read the stack, find the channel dimension of size=2, 3, or 4,
      split into separate TIFF files (channel_1.tif, channel_2.tif, channel_3.tif,
      channel_4.tif as applicable), and save them in the new subfolder.
    """
    input_dir = os.path.abspath(input_dir)

    # Make sure the directory exists
    if not os.path.isdir(input_dir):
        print(f"ERROR: The provided directory does not exist: {input_dir}")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif'):
            tif_path = os.path.join(input_dir, filename)

            # Separate name & extension
            name_no_ext, ext = os.path.splitext(filename)

            # Remove whitespace from base name
            name_no_ws = name_no_ext.replace(" ", "")

            # Also remove whitespace from the entire filename
            new_filename = f"{name_no_ws}{ext}"

            # Create a subfolder named after the "whitespace-removed" base name
            subfolder_path = os.path.join(input_dir, name_no_ws)
            os.makedirs(subfolder_path, exist_ok=True)

            # Full path to the new copy of the TIFF
            new_tif_path = os.path.join(subfolder_path, new_filename)

            # Copy the original TIFF into the new subfolder (with removed whitespace in name)
            shutil.copy2(tif_path, new_tif_path)

            # Read the stack
            stack = tifffile.imread(new_tif_path)
            shape = stack.shape

            # Find the dimension that has size=2, 3, or 4 (channel dimension)
            channel_axis = None
            channel_size = None
            for axis, size in enumerate(shape):
                if size in (2, 3, 4):
                    channel_axis = axis
                    channel_size = size
                    break

            if channel_axis is None:
                print(f"WARNING: No channel dimension of size=2, 3, or 4 found for {filename}. Skipping.")
                continue

            # Split into channels
            for i in range(channel_size):
                channel_data = stack.take(i, axis=channel_axis)
                channel_path = os.path.join(subfolder_path, f"channel_{i+1}.tif")
                tifffile.imwrite(channel_path, channel_data)

            # Print summary
            channel_list = [f"channel_{i+1}.tif" for i in range(channel_size)]
            print(f"Processed '{filename}' into subfolder '{name_no_ws}':")
            print(f"  -> Original copied to {new_filename}")
            print(f"  -> Created {', '.join(channel_list)}")

def main():
    parser = argparse.ArgumentParser(
        description="Split 2-, 3-, or 4-channel TIFF stacks into separate files, removing whitespace from file/subfolder names."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the TIF stacks."
    )

    args = parser.parse_args()
    split_channels_in_directory(args.input_dir)

if __name__ == "__main__":
    main()





