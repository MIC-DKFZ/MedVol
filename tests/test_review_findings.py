"""Regression tests for review findings deferred to the conventions rework (mlarray FINDINGS.md).

H12, M15, M16 and M18 concern NRRD and are covered in tests/test_nrrd.py.
"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from mlarray import MLArray

import medvol
import phantom


class TestDeferredFindings(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_h10_arrays_with_a_channel_axis_can_be_exported(self):
        truth = phantom.Truth()
        nifti, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", truth))
        data = np.stack([nifti.array, nifti.array + 1], axis=-1)            # (z, y, x, c)
        image = MLArray(data, axes=["z", "y", "x", {"name": "c", "type": "channel"}], spaces=[nifti.space().to_dict()])
        out = self.dir / "c.nii.gz"
        report = medvol.write(medvol.Image.from_mlarray(image), out)
        self.assertEqual(nib.load(str(out)).shape, (5, 6, 7, 2))
        self.assertTrue(any("NIfTI dim 4" in w for w in report.warnings))
        again, _ = medvol.read(out)
        self.assertTrue(np.array_equal(again.array[0], nifti.array))     # first channel / time point

    def test_h11_axis_order_is_explicit_so_zyx_arrays_keep_their_geometry(self):
        truth = phantom.Truth()
        nifti, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", truth))
        self.assertEqual([a.name for a in nifti.meta.axes], ["z", "y", "x"])
        medvol.write(nifti, self.dir / "a.mla")
        for target in ("b.nii.gz", "b.mgz"):
            medvol.convert(self.dir / "a.mla", self.dir / target)
            again, _ = medvol.read(self.dir / target)
            self.assertLess(phantom.world_error(again, truth), 1e-4, target)

    def test_m14_the_world_frame_is_explicit_not_a_per_axis_code(self):
        image, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz"))
        space = image.space()
        self.assertEqual([a.increases_toward for a in space.axes], ["left", "posterior", "superior"])
        self.assertEqual(space.letter_codes(), ("R", "I", "P"))     # array-axis directions, derived only

    def test_m17_nifti_header_fields_are_written_back(self):
        extra = {"descrip": "kept", "intent_code": 2, "intent_p1": 3.5, "intent_name": "ttest", "cal_max": 9.0}
        path = phantom.write_nifti(self.dir / "a.nii.gz", extra=extra, qcode=2, scode=3)
        medvol.convert(path, self.dir / "a.mla")
        medvol.convert(self.dir / "a.mla", self.dir / "b.nii.gz")
        header = nib.load(str(self.dir / "b.nii.gz")).header
        self.assertEqual((header["descrip"].item(), int(header["intent_code"]), float(header["intent_p1"]),
                          header["intent_name"].item(), float(header["cal_max"])), (b"kept", 2, 3.5, b"ttest", 9.0))
        self.assertEqual((int(header["qform_code"]), int(header["sform_code"])), (2, 3))

    def test_l11_scaled_data_keeps_its_stored_dtype(self):
        truth = phantom.Truth(dtype="int16")
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, extra={"scl_slope": 0.5, "scl_inter": 10.0})
        medvol.convert(path, self.dir / "a.mla")
        image = MLArray(self.dir / "a.mla")
        self.assertEqual(image.dtype, np.int16)
        self.assertEqual((image.values.scale, image.values.offset), (0.5, 10.0))


if __name__ == "__main__":
    unittest.main()
