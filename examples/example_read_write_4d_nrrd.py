from medvol import MedVol
import numpy as np
import os


def example_read_write_4d(load_filepath, save_filepath):
    # Read 4D image
    image = MedVol(load_filepath)

    spacing1 = image.spacing
    origin1 = image.origin
    direction1 = image.direction
    affine1 = image.affine

    # Write as new 4D image
    image.save(save_filepath)

    # Read new 4D image
    image = MedVol(save_filepath)

    spacing2 = image.spacing
    origin2 = image.origin
    direction2 = image.direction
    affine2 = image.affine

    # Check if metadata is still correct
    assert np.array_equal(spacing1, spacing2)
    assert np.array_equal(origin1, origin2)
    assert np.array_equal(direction1, direction2)
    assert np.array_equal(affine1, affine2)

    print("All checks passed.")

    os.remove(save_filepath)


if __name__ == '__main__':
    load_filepath = "examples/data/4d_img.nrrd"
    save_filepath = "examples/data/4d_img_tmp.nrrd"

    example_read_write_4d(load_filepath, save_filepath)