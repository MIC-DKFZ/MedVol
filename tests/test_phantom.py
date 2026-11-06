"""The phantom generators themselves agree with the ground truth (decoded with reference libraries)."""

import glob
import tempfile
import unittest
from pathlib import Path

import numpy as np

import phantom


def _error(values, index_to_lps, truth):
    index = np.indices(values.shape).reshape(values.ndim, -1)
    world = (index_to_lps @ np.vstack([index[:3], np.ones(index.shape[1])]))[:3].T
    return float(np.max(np.linalg.norm(world - truth.world_of_values(values.reshape(-1)), axis=1)))


class TestPhantom(unittest.TestCase):
    def test_nifti_forms_and_codes(self):
        import nibabel as nib

        truth = phantom.Truth()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = phantom.write_nifti(Path(tmpdir) / "a.nii.gz", truth, qcode=2, scode=4)
            image = nib.load(path)
            self.assertEqual((int(image.header["qform_code"]), int(image.header["sform_code"])), (2, 4))
            for form in (image.header.get_qform(), image.header.get_sform()):
                self.assertLess(_error(np.asarray(image.dataobj), phantom.FLIP4 @ form, truth), 1e-4)
            path = phantom.write_nifti(Path(tmpdir) / "b.nii", truth, qform=None)
            self.assertEqual(int(nib.load(path).header["qform_code"]), 0)

    def test_nrrd_spaces(self):
        import nrrd

        truth = phantom.Truth()
        with tempfile.TemporaryDirectory() as tmpdir:
            for space, to_space in phantom.NRRD_FRAMES.items():
                with self.subTest(space=space):
                    data, header = nrrd.read(str(phantom.write_nrrd(Path(tmpdir) / f"{space}.nrrd", truth, space)))
                    matrix = np.eye(4)
                    matrix[:3, :3] = np.linalg.inv(to_space) @ header["space directions"].T
                    matrix[:3, 3] = np.linalg.inv(to_space) @ header["space origin"]
                    self.assertLess(_error(data, matrix, truth), 1e-4)

    def test_dicom_series_positions(self):
        import pydicom

        for truth in (phantom.Truth(dtype="int16"), phantom.sheared_truth()):
            truth.dtype = "int16"
            with tempfile.TemporaryDirectory() as tmpdir:
                phantom.write_dicom_series(tmpdir, truth)
                worst = 0.0
                for path in glob.glob(f"{tmpdir}/*.dcm"):
                    ds = pydicom.dcmread(path)
                    iop = np.array(ds.ImageOrientationPatient, float)
                    ipp = np.array(ds.ImagePositionPatient, float)
                    rs, cs = map(float, ds.PixelSpacing)
                    frame = ds.pixel_array  # [row j, column i]
                    j, i = np.indices(frame.shape).reshape(2, -1)
                    world = ipp + np.outer(i * cs, iop[:3]) + np.outer(j * rs, iop[3:])
                    true = truth.world_of_values(frame.reshape(-1))
                    worst = max(worst, float(np.max(np.linalg.norm(world - true, axis=1))))
                self.assertLess(worst, 1e-4)


if __name__ == "__main__":
    unittest.main()
