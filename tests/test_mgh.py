import gzip
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

import medvol
import phantom
import phantom_mgh
from medvol import registry
from medvol.plugins.mgh import MghPlugin
from medvol.report import LossError

registry.register(MghPlugin())

#: mri_info --vox2ras (FreeSurfer 7.4.1) of a (5, 6, 7) .mgz with goodRASFlag = 0
#: (KB experiments/verify_freesurfer_afni/v01_freesurfer.out.txt, section I)
FREESURFER_GOODRAS0 = np.array([[-1.0, 0, 0, 2.5], [0, 0, 1, -3.5], [0, -1, 0, 3], [0, 0, 0, 1]])


def _bytes(path):
    opener = gzip.open if str(path).endswith(".mgz") else open
    with opener(str(path), "rb") as handle:
        return handle.read()


def _header(path):
    image, _ = medvol.read(path)
    return dict(image.meta.source[0].header)


class MghCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def round_trip(self, path, **options):
        """X -> .mla -> X; returns the path of the rewritten file."""
        image, _ = medvol.read(path, **options)
        stem = Path(path).name.split(".")[0]
        mla = self.dir / f"{stem}.mla"
        medvol.write(image, mla)
        again, _ = medvol.read(mla)
        out = self.dir / f"{stem}_rt{Path(path).suffix}"
        medvol.write(again, out, **options)
        return out


class TestRead(MghCase):
    def test_world_positions_mgh_and_mgz(self):
        truth = phantom.Truth()
        for name in ("a.mgh", "a.mgz"):
            with self.subTest(file=name):
                path = phantom.write_mgh(self.dir / name, truth)
                image, report = medvol.read(path)
                self.assertEqual(report.format, "mgh")
                self.assertLess(phantom.world_error(image, truth), 1e-4)
                self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
                self.assertEqual(image.meta.source[0].index_map["axis_order"], (2, 1, 0))
                self.assertEqual(image.array.dtype, np.float32)
                self.assertTrue(image.array.dtype.isnative)

    def test_ras_world_frame_is_a_row_sign_change(self):
        truth = phantom.Truth()
        path = phantom.write_mgh(self.dir / "a.mgz", truth)
        lps, _ = medvol.read(path)
        ras, _ = medvol.read(path, world_frame="ras")
        np.testing.assert_allclose(ras.space().matrix(), np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps.space().matrix())
        self.assertLess(phantom.world_error(ras, truth), 1e-4)

    def test_c_ras_is_the_world_point_at_n_over_2(self):
        truth = phantom.Truth()        # odd sizes 5 and 7: N/2 differs from (N - 1)/2 by half a voxel
        path = phantom_mgh.write_truth(self.dir / "a.mgz", truth)
        image, _ = medvol.read(path)
        self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_header_is_readable_and_ordered(self):
        path = phantom_mgh.write_truth(self.dir / "a.mgz", dof=7)
        header = _header(path)
        self.assertEqual(list(header), ["version", "dims", "type", "dof", "goodRASFlag", "delta", "Mdc", "Pxyz_c",
                                        "tr", "flip_angle", "te", "ti", "fov"])
        self.assertEqual(header["delta"], (0.5, 1.0, 2.0))
        self.assertEqual(header["dof"], 7)
        self.assertEqual(len(header["Mdc"]), 3)


class TestGoodRasFlag(MghCase):
    def test_flag_0_follows_freesurfer_not_nibabel(self):
        truth = phantom.Truth()
        delta, mdc, c_ras = phantom_mgh.fields_from_truth(truth)
        path = phantom_mgh.write_raw(self.dir / "bad.mgz", truth.volume().astype(np.float32), delta, mdc, c_ras,
                                     good_ras=0)
        image, report = medvol.read(path, world_frame="ras")
        self.assertIn("goodRASFlag = 0", report.warnings[0])
        self.assertIn("FreeSurfer default", report.choices["mgh_geometry"])
        # the space maps array axes ["x", "y", "z"] = file i, j, k: compare with mri_info directly
        self.assertEqual(list(image.space().from_array.array_axes), ["x", "y", "z"])
        np.testing.assert_array_equal(image.space().matrix(), FREESURFER_GOODRAS0)
        self.assertFalse(np.allclose(nib.load(str(path)).affine, FREESURFER_GOODRAS0))   # nibabel 5.4.2: LSP

    def test_flag_0_round_trip_keeps_the_stored_fields(self):
        truth = phantom.Truth()
        delta, mdc, c_ras = phantom_mgh.fields_from_truth(truth)
        path = phantom_mgh.write_raw(self.dir / "bad.mgz", truth.volume().astype(np.float32), delta, mdc, c_ras,
                                     good_ras=0)
        out = self.round_trip(path)
        self.assertEqual(_bytes(out), _bytes(path))

    def test_flag_0_regenerated_is_explicit_lia(self):
        data = np.zeros((5, 6, 7), np.float32)
        path = phantom_mgh.write_raw(self.dir / "bad.mgz", data, [3.0, 3.0, 3.0], np.eye(3), [9.0, 9.0, 9.0],
                                     good_ras=0)
        image, _ = medvol.read(path)
        out = self.dir / "regen.mgz"
        with mock.patch.object(MghPlugin, "_geometry_unchanged", return_value=False):
            report = medvol.write(image, out)
        self.assertTrue(any("goodRASFlag = 1" in w for w in report.warnings))
        header = nib.load(str(out)).header
        self.assertEqual(int(header["goodRASFlag"]), 1)
        np.testing.assert_array_equal(nib.load(str(out)).affine, FREESURFER_GOODRAS0)


class TestRoundTrip(MghCase):
    def test_same_format_round_trip_is_exact(self):
        truth = phantom.Truth()
        cases = {
            "nib.mgz": lambda p: phantom.write_mgh(p, truth),
            "nib.mgh": lambda p: phantom.write_mgh(p, truth),
            "int16.mgz": lambda p: phantom_mgh.write_truth(p, truth, dtype="int16"),
            "uint8.mgh": lambda p: phantom_mgh.write_truth(p, phantom.Truth(shape=(4, 3, 2)), dtype="uint8"),
            "nofooter.mgz": lambda p: phantom_mgh.write_truth(p, truth, footer=None),
            "frames.mgz": lambda p: phantom_mgh.write_truth(p, phantom.Truth(shape=(5, 6, 7, 3)),
                                                            footer=(2000.0, 0.2, 30.0, 0.0, 0.0)),
            "sheared.mgz": lambda p: phantom.write_mgh(p, phantom.sheared_truth()),
        }
        for name, make in cases.items():
            with self.subTest(file=name):
                path = make(self.dir / name)
                out = self.round_trip(path)
                self.assertEqual(_header(out), _header(path))
                self.assertEqual(_bytes(out), _bytes(path))
                self.assertEqual(out.suffix, path.suffix)
                with open(out, "rb") as handle:
                    self.assertEqual(handle.read(2) == b"\x1f\x8b", name.endswith(".mgz"))

    def test_scan_parameters_tags_and_unused_bytes_are_kept(self):
        footer = (2300.0, 0.15707964, 2.95, 900.0, 256.0)
        tags = b"\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x10talairach.xfm\x00\x00\x00"
        path = phantom_mgh.write_truth(self.dir / "a.mgz", footer=footer, tags=tags, unused=b"\x01\x02")
        header = _header(path)
        self.assertEqual([header[k] for k in ("tr", "flip_angle", "te", "ti", "fov")], list(footer))
        self.assertEqual(header["tags"], tags)
        self.assertEqual(header["unused"][:2], b"\x01\x02")
        out = self.round_trip(path)
        self.assertEqual(_bytes(out), _bytes(path))
        nib_header = nib.load(str(out)).header
        self.assertEqual(float(nib_header["te"]), float(np.float32(2.95)))

    def test_core_owned_fields_regenerate_to_the_original_values(self):
        """Spec 4: regenerating the core-owned fields from an unchanged core reproduces them."""
        cases = {
            "plain": lambda p: phantom.write_mgh(p, phantom.Truth()),
            "handmade": lambda p: phantom_mgh.write_truth(p, phantom.Truth(), dtype="int32"),
            "frames": lambda p: phantom_mgh.write_truth(p, phantom.Truth(shape=(5, 6, 7, 3)),
                                                        footer=(2000.0, 0.2, 30.0, 0.0, 0.0)),
            "large_origin": lambda p: phantom.write_mgh(p, phantom.Truth(affine_lps=_shifted(1234.5678, -2345.6789, 3456.7891))),
        }
        for name, make in cases.items():
            with self.subTest(case=name):
                path = make(self.dir / f"{name}.mgz")
                image, _ = medvol.read(path)
                out = self.dir / f"{name}_regen.mgz"
                with mock.patch.object(MghPlugin, "_geometry_unchanged", return_value=False):
                    report = medvol.write(image, out)
                self.assertNotIn("geometry", report.choices)
                original, regenerated = _header(path), _header(out)
                for key in ("dims", "type", "goodRASFlag", "delta", "Mdc", "Pxyz_c", "tr"):
                    self.assertEqual(regenerated[key], original[key], msg=key)
                self.assertEqual(_bytes(out), _bytes(path))


def _shifted(*offset):
    affine = phantom.Truth().affine_lps.copy()
    affine[:3, 3] += offset
    return affine


class TestFrames(MghCase):
    def test_frames_with_tr_are_time(self):
        truth = phantom.Truth(shape=(5, 6, 7, 3))
        path = phantom_mgh.write_truth(self.dir / "a.mgz", truth, footer=(2000.0, 0.0, 0.0, 0.0, 0.0))
        image, _ = medvol.read(path)
        self.assertEqual([(a.name, a.type) for a in image.meta.axes],
                         [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")])
        self.assertEqual((image.meta.axes[0].step, image.meta.axes[0].unit), (2000.0, "millisecond"))
        self.assertEqual(image.meta.source[0].index_map["axis_order"], (3, 2, 1, 0))
        self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_frames_without_tr_are_other(self):
        truth = phantom.Truth(shape=(5, 6, 7, 2))
        for name, footer in (("zero_tr.mgz", (0.0,) * 5), ("no_footer.mgz", None)):
            with self.subTest(file=name):
                image, _ = medvol.read(phantom_mgh.write_truth(self.dir / name, truth, footer=footer))
                self.assertEqual((image.meta.axes[0].name, image.meta.axes[0].type), ("f", "other"))
                self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_nifti_time_in_seconds_becomes_tr_in_milliseconds(self):
        truth = phantom.Truth(shape=(5, 6, 7, 3))
        nifti = phantom.write_nifti(self.dir / "a.nii.gz", truth)     # pixdim[4] = 1, unit second
        image, _ = medvol.read(nifti)
        self.assertEqual((image.meta.axes[0].step, image.meta.axes[0].unit), (1.0, "second"))
        out = self.dir / "a.mgz"
        medvol.write(image, out)
        self.assertEqual(float(nib.load(str(out)).header["tr"]), 1000.0)
        again, _ = medvol.read(out)
        self.assertEqual((again.meta.axes[0].type, again.meta.axes[0].step), ("time", 1000.0))
        self.assertLess(phantom.world_error(again, truth), 1e-4)


class TestShear(MghCase):
    def test_shear_is_kept_exactly_with_a_warning(self):
        truth = phantom.sheared_truth()
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, qform=phantom.Truth().affine_ras)
        image, _ = medvol.read(path)
        out = self.dir / "a.mgz"
        report = medvol.write(image, out)
        self.assertEqual(report.dropped, [])
        self.assertTrue(any("non-orthonormal Mdc" in w for w in report.warnings))
        again, _ = medvol.read(out)
        self.assertLess(phantom.world_error(again, truth), 1e-4)
        medvol.write(image, self.dir / "b.mgz", on_loss="error")   # nothing is lost


class TestCrossFormat(MghCase):
    def test_nifti_to_mgz_keeps_world_positions_and_no_nifti_fields(self):
        truth = phantom.Truth()
        nifti = phantom.write_nifti(self.dir / "a.nii.gz", truth, extra={"descrip": "phantom", "cal_max": 9.0})
        image, _ = medvol.read(nifti)
        out = self.dir / "a.mgz"
        report = medvol.write(image, out)
        self.assertEqual(report.dropped, [])
        again, _ = medvol.read(out)
        self.assertLess(phantom.world_error(again, truth), 1e-4)
        header = dict(again.meta.source[-1].header)
        self.assertEqual(list(header), ["version", "dims", "type", "dof", "goodRASFlag", "delta", "Mdc", "Pxyz_c",
                                        "tr", "flip_angle", "te", "ti", "fov"])
        self.assertEqual(header["type"], 1)                              # int32 kept
        self.assertEqual((header["tr"], header["dof"]), (0.0, 0))
        self.assertEqual(len(_bytes(out)), 284 + 4 * 5 * 6 * 7 + 20)   # no stray bytes (tags, extensions)

    def test_mgz_to_nifti_keeps_world_positions(self):
        truth = phantom.Truth()
        image, _ = medvol.read(phantom.write_mgh(self.dir / "a.mgz", truth))
        out = self.dir / "a.nii.gz"
        medvol.write(image, out)
        again, _ = medvol.read(out)
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_value_scaling_is_reported(self):
        truth = phantom.Truth(dtype="int16")
        image, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", truth, extra={"scl_slope": 2.0}))
        report = medvol.write(image, self.dir / "a.mgz")
        self.assertTrue(any(d.startswith("values.scale") for d in report.dropped))

    def test_unsupported_dtype_is_refused(self):
        image, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", phantom.Truth(dtype="float64")))
        with self.assertRaises(LossError):
            medvol.write(image, self.dir / "a.mgz")

    def test_sniff(self):
        plugin = MghPlugin()
        self.assertEqual(plugin.sniff(phantom.write_mgh(self.dir / "a.mgz")), 1.0)
        self.assertEqual(plugin.sniff(phantom.write_nifti(self.dir / "a.nii.gz")), 0.0)
        (self.dir / "junk.mgh").write_bytes(b"not an mgh file")
        self.assertEqual(plugin.sniff(self.dir / "junk.mgh"), 0.0)


if __name__ == "__main__":
    unittest.main()
