"""Synthetic bounding-box tests; no patient files, weights, or training."""
import numpy as np
from preprocess_ct import find_3d_bbox, expand_bbox_3d


def main():
    shape = (400, 400, 400)
    observed = []
    for side in (10, 20, 30, 50):
        mask = np.zeros(shape, dtype=np.uint8)
        start = 160
        mask[start:start+side, start:start+side, start:start+side] = 1
        bbox = find_3d_bbox(mask)
        z1, z2, y1, y2, x1, x2 = expand_bbox_3d(bbox, shape, context_scale=1.8)
        crop_lengths = (z2-z1, y2-y1, x2-x1)
        expected = int(round(side*1.8))
        assert crop_lengths == (expected, expected, expected), (side, crop_lengths)
        relative_width = side / crop_lengths[2]
        assert abs(relative_width - 1/1.8) < 0.03
        observed.append((side, crop_lengths, relative_width))
        del mask
    # Border handling: should stay inside original array bounds.
    near_edge = (0, 10, 0, 10, 0, 10)
    box = expand_bbox_3d(near_edge, shape)
    assert all(0 <= b <= n for b, n in zip(box, (400,400,400,400,400,400)))
    assert all(box[i] < box[i+1] for i in (0,2,4))
    print("PASS proportional 1.8x crop, no minimum crop-size constraint")
    for side, lengths, frac in observed:
        print(f"  ROI {side} voxels -> crop {lengths} voxels; fraction={frac:.4f}")

if __name__ == "__main__":
    main()
