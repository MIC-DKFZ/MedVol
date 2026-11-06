import tempfile
import unittest
from pathlib import Path

import numpy as np

import medvol
import phantom
from medvol.geometry import GeometryError


class TestReorient(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_closest_layout_keeps_every_voxel_in_place(self):
        truth = phantom.Truth()
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth)
        for target, words in (("LPS", ("L", "P", "S")), ("RAS", ("R", "A", "S"))):
            with self.subTest(target=target):
                image, report = medvol.read(path, reorient=target)
                self.assertLess(phantom.world_error(image, truth), 1e-4)
                space = image.space()
                names = [a.name for a in image.meta.axes]
                fastest_first = [space.letter_codes()[list(space.from_array.array_axes).index(n)] for n in names[::-1]]
                if target == "RAS":
                    image_ras, _ = medvol.read(path, reorient=target, world_frame="ras")
                    self.assertLess(phantom.world_error(image_ras, truth), 1e-4)
                    space = image_ras.space()
                    fastest_first = [space.letter_codes()[list(space.from_array.array_axes).index(n)] for n in names[::-1]]
                self.assertEqual(tuple(fastest_first), words)
                self.assertIn(target, report.choices["reorient"])
                self.assertIn("reorient", image.meta.history[0]["settings"])

    def test_index_map_is_composed_and_writers_regenerate_geometry(self):
        truth = phantom.Truth()
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth)
        image, _ = medvol.read(path, reorient="LPS")
        index_map = image.meta.source[0].index_map
        self.assertEqual(sorted(index_map["axis_order"]), [0, 1, 2])
        self.assertTrue(any(index_map["flipped"]))
        for name in ("b.nii.gz", "b.nrrd", "b.mha", "b.mgz", "b.mla"):
            medvol.write(image, self.dir / name)
            again, _ = medvol.read(self.dir / name)
            self.assertLess(phantom.world_error(again, truth), 1e-4, name)

    def test_axis_aligned_lps_data_is_left_alone(self):
        truth = phantom.Truth(affine_lps=np.diag([2.0, 1.0, 0.5, 1.0]))
        image, report = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", truth), reorient="LPS")
        original, _ = medvol.read(self.dir / "a.nii.gz")
        np.testing.assert_array_equal(image.array, original.array)
        self.assertIn("already", report.choices["reorient"])

    def test_ambiguous_orientation_is_refused(self):
        rotation = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], [np.sin(np.pi / 4), np.cos(np.pi / 4), 0], [0, 0, 1]])
        affine = np.eye(4)
        affine[:3, :3] = rotation
        path = phantom.write_nifti(self.dir / "a.nii.gz", phantom.Truth(affine_lps=affine))
        with self.assertRaises(GeometryError):
            medvol.read(path, reorient="LPS")


if __name__ == "__main__":
    unittest.main()
