from medvol import MedVol
import numpy as np


def print_affine():
    print(MedVol("examples/data/2d_img.nii.gz").affine)
    print(MedVol("examples/data/3d_img.nii.gz").affine)
    print(MedVol("examples/data/4d_img.nii.gz").affine)
    print(MedVol("examples/data/4d_img.nrrd").affine)
    print(MedVol(np.zeros((50, 50, 50))).affine)


if __name__ == '__main__':
    print_affine()