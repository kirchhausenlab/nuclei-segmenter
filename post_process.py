import os
import argparse

import numpy as np
import tifffile as tiff
from scipy.ndimage import label as ndi_label, binary_fill_holes
from skimage.morphology import convex_hull_image, closing, disk
from skimage.measure import label as sk_label, regionprops
from skimage.color import label2rgb
import cv2

# Parameters for post‐processing
MIN_SIZE = 1500
MIN_EDGE_SIZE = 100
MIN_SOLIDITY = 0.4

def remove_small_objects(mask, min_size):
    processed = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        labeled_plane, num_features = ndi_label(mask[z])
        for i in range(1, num_features + 1):
            pix = (labeled_plane == i)
            size = np.sum(pix)
            if size >= min_size:
                processed[z][pix] = 1
            else:
                ys, xs = np.where(pix)
                if (0 in ys or mask[z].shape[0]-1 in ys or
                    0 in xs or mask[z].shape[1]-1 in xs) and size >= MIN_EDGE_SIZE:
                    processed[z][pix] = 1
    return processed

def refine_solidity(mask, min_solidity):
    refined = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        labeled_plane, num_features = ndi_label(mask[z])
        for i in range(1, num_features + 1):
            pix = (labeled_plane == i)
            filled = binary_fill_holes(pix)
            hull = convex_hull_image(filled)
            solidity = np.sum(filled) / np.sum(hull)
            print(f"Plane {z}, Object {i}, Solidity: {solidity:.3f}")
            if solidity >= min_solidity:
                refined[z][filled] = 1
    return refined

def smooth_boundaries(mask):
    smoothed = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        smoothed[z] = closing(mask[z], disk(2))
    return smoothed

def interpolate_missing_objects(mask):
    interp = mask.copy()
    for z in range(1, mask.shape[0] - 1):
        prev_f = mask[z - 1]
        next_f = mask[z + 1]
        overlap = (prev_f & next_f) > 0
        miss = (mask[z] == 0) & overlap
        interp[z][miss] = (prev_f[miss] + next_f[miss]) // 2
    return interp

def post_process(mask):
    mask = remove_small_objects(mask, MIN_SIZE)
    mask = refine_solidity(mask, MIN_SOLIDITY)
    mask = smooth_boundaries(mask)
    mask = interpolate_missing_objects(mask)
    return mask

def process_combined_masks(source_dir):
    for root, _, files in os.walk(source_dir):
        if 'combined_masks.tif' in files:
            inp = os.path.join(root, 'combined_masks.tif')
            out = os.path.join(root, 'processed_masks.tif')
            mask = tiff.imread(inp)
            proc = post_process(mask)
            tiff.imwrite(out, proc, dtype=proc.dtype)
            print(f"Processed and saved: {out}")

def track_objects(masks):
    tracked = np.zeros_like(masks, dtype=np.int32)
    cid = 0
    for z in range(masks.shape[0]):
        curr = sk_label(masks[z])
        props = regionprops(curr)
        if z == 0:
            for p in props:
                cid += 1
                tracked[z][curr == p.label] = cid
        else:
            prev = tracked[z - 1]
            for p in props:
                mask_p = (curr == p.label)
                oids = np.unique(prev[mask_p])
                oids = oids[oids != 0]
                if oids.size:
                    tracked[z][mask_p] = np.min(oids)
                else:
                    cid += 1
                    tracked[z][mask_p] = cid
    return tracked

def remove_single_plane_objects(tracked):
    clean = tracked.copy()
    for lbl in np.unique(clean)[1:]:
        planes = np.where(np.any(clean == lbl, axis=(1,2)))[0]
        if planes.size == 1:
            clean[clean == lbl] = 0
    return clean

def make_key(channel, tracked, out_path="key.tif", font_scale=1.2):
    chan = channel.astype(float)
    if chan.max() > 0:
        chan /= chan.max()
    zs, h, w = chan.shape
    out = np.zeros((zs, h, w, 3), dtype=np.uint8)
    for z in range(zs):
        overlay = label2rgb(tracked[z], image=chan[z], bg_label=0, alpha=0.5)
        plane = (overlay * 255).astype(np.uint8)
        for p in regionprops(tracked[z]):
            y, x = map(int, p.centroid)
            cv2.putText(plane, str(p.label), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255,0,0), 2, cv2.LINE_AA)
        out[z] = plane
    tiff.imwrite(out_path, out, photometric='rgb')
    print(f"Saved key TIF to: {out_path}")

def process_directory(base_dir):
    for root, _, files in os.walk(base_dir):
        if 'processed_masks.tif' in files and 'channel_1.tif' in files:
            mp = os.path.join(root, 'processed_masks.tif')
            cp = os.path.join(root, 'channel_1.tif')
            masks = tiff.imread(mp)
            chan1 = tiff.imread(cp)

            tracked = track_objects(masks)
            tracked = remove_single_plane_objects(tracked)

            out_tm = os.path.join(root, 'tracked_masks.tif')
            tiff.imwrite(out_tm, tracked.astype(np.uint16))
            print(f"Saved tracked masks to {out_tm}")

            keyp = os.path.join(root, 'key.tif')
            make_key(chan1, tracked, out_path=keyp, font_scale=1.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Post-process combined masks and then track objects.'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing combined_masks.tif and channel_1.tif')
    args = parser.parse_args()

    process_combined_masks(args.input_dir)
    process_directory(args.input_dir)


