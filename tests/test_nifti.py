import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
from mlarray import MLArray

import medvol
import phantom
from medvol.geometry import GeometryError
from medvol.plugins.nifti import NiftiPlugin


def _header_bytes(path):
    header = nib.load(str(path)).header
    return {key: np.asarray(header[key]).tobytes() for key in header.keys()}


def _data(path):
    return np.asarray(nib.load(str(path)).dataobj.get_unscaled())


class NiftiCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def round_trip(self, path, **options):
        """X -> .mla -> X; returns the path of the rewritten file."""
        image, _ = medvol.read(path, **options)
        mla = self.dir / (Path(path).name.split(".")[0] + ".mla")
        medvol.write(image, mla)
        again, _ = medvol.read(mla)
        suffix = "".join(Path(path).suffixes)
        out = self.dir / (Path(path).name.split(".")[0] + "_rt" + suffix)
        medvol.write(again, out, **options)
        return out


class TestRead(NiftiCase):
    def test_world_positions_of_all_form_variants(self):
        truth = phantom.Truth()
        variants = {
            "both": {}, "codes_2_4": {"qcode": 2, "scode": 4}, "qform_only": {"sform": None},
            "sform_only": {"qform": None}, "nifti2": {"version": 2},
        }
        for name, kwargs in variants.items():
            with self.subTest(variant=name):
                path = phantom.write_nifti(self.dir / f"{name}.nii.gz", truth, **kwargs)
                image, _ = medvol.read(path)
                self.assertLess(phantom.world_error(image, truth), 1e-4)
                self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
                self.assertEqual(image.meta.source[0].index_map["axis_order"], (2, 1, 0))

    def test_space_names_follow_the_codes(self):
        path = phantom.write_nifti(self.dir / "a.nii.gz", qcode=1, scode=4)
        image, _ = medvol.read(path)
        self.assertEqual(image.space().name, "mni")
        self.assertEqual(image.space().reference["id"], "MNI152")
        image, _ = medvol.read(path, nifti_form="qform")
        self.assertEqual(image.space().name, "patient")

    def test_disagreeing_forms_follow_the_policy(self):
        truth = phantom.Truth()
        shifted = truth.affine_ras.copy()
        shifted[:3, 3] += [0, 0, 4.9]
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, qform=shifted)
        image, report = medvol.read(path)                                     # nibabel rule: sform
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assertIn("disagree by 4.9", report.warnings[0])
        self.assertIn("disagree by 4.9", image.meta.history[0]["warnings"][0])
        with self.assertRaises(GeometryError):
            medvol.read(path, nifti_form_disagreement="error")
        image, _ = medvol.read(path, nifti_form="qform")
        self.assertAlmostEqual(phantom.world_error(image, truth), 4.9, places=4)

    def test_files_without_codes_need_a_policy(self):
        truth = phantom.Truth()
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, qform=None, sform=None)
        with self.assertRaises(GeometryError):
            medvol.read(path)
        expected = {
            "spec": np.diag([0.5, 1.0, 2.0, 1.0]),
            "itk": np.diag([-0.5, -1.0, 2.0, 1.0]),
            "nibabel": nib.load(str(path)).header.get_base_affine(),
        }
        for rule, ras in expected.items():
            with self.subTest(rule=rule):
                image, report = medvol.read(path, nifti_no_codes=rule)
                lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ ras
                np.testing.assert_allclose(image.space().matrix()[:, [0, 1, 2, 3]], lps, atol=1e-6)
                self.assertEqual(report.choices["nifti_no_codes"], rule)

    def test_ras_world_frame_is_a_row_sign_change(self):
        truth = phantom.Truth()
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth)
        lps, _ = medvol.read(path)
        ras, _ = medvol.read(path, world_frame="ras")
        self.assertEqual([a.increases_toward for a in ras.space().axes], ["right", "anterior", "superior"])
        np.testing.assert_allclose(ras.space().matrix(), np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps.space().matrix())
        self.assertLess(phantom.world_error(ras, truth), 1e-4)
        np.testing.assert_array_equal(ras.array, lps.array)

    def test_time_and_vector_axes(self):
        truth = phantom.Truth(shape=(5, 6, 7, 3, 2))
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, extra={"intent_code": 1007})
        image, _ = medvol.read(path)
        self.assertEqual([(a.name, a.type) for a in image.meta.axes],
                         [("c", "component"), ("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")])
        self.assertEqual(image.meta.axes[1].unit, "second")
        self.assertLess(phantom.world_error(image, truth), 1e-4)


class TestRoundTrip(NiftiCase):
    def test_same_format_round_trip_is_exact(self):
        truth = phantom.Truth()
        cases = {
            "both.nii.gz": {}, "codes.nii": {"qcode": 2, "scode": 4}, "qonly.nii.gz": {"sform": None},
            "v2.nii.gz": {"version": 2}, "time.nii.gz": {"truth": phantom.Truth(shape=(5, 6, 7, 3))},
            "sheared.nii.gz": {"truth": phantom.sheared_truth(), "qform": phantom.Truth().affine_ras},
            "pair.hdr": {},
        }
        for name, kwargs in cases.items():
            with self.subTest(file=name):
                kwargs = dict(kwargs)
                path = phantom.write_nifti(self.dir / name, kwargs.pop("truth", truth), **kwargs)
                out = self.round_trip(path)
                self.assertEqual(_header_bytes(out), _header_bytes(path))
                np.testing.assert_array_equal(_data(out), _data(path))

    def test_scaling_extensions_and_text_survive(self):
        truth = phantom.Truth(dtype="int16")
        extra = {"scl_slope": 2.0, "scl_inter": -1024.0, "descrip": "phantom 0.7", "cal_min": 1.5, "cal_max": 99.25}
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, extra=extra,
                                   extensions=[(6, b"comment\x00"), (40, bytes(range(10)))])

        read, _ = medvol.read(path)
        self.assertEqual((read.meta.values.scale, read.meta.values.offset), (2.0, -1024.0))
        self.assertEqual(read.array.dtype, np.int16)
        self.assertEqual(read.meta.source[0].header["descrip"], "phantom 0.7")
        self.assertEqual(read.meta.source[0].header["extensions"][1]["content"], bytes(range(10)))
        out = self.round_trip(path)
        self.assertEqual(_header_bytes(out), _header_bytes(path))
        extensions = [[(e.get_code(), e.content) for e in nib.load(str(p)).header.extensions] for p in (out, path)]
        self.assertEqual(extensions[0], extensions[1])
        self.assertEqual(len(extensions[0]), 2)

    def test_core_owned_fields_regenerate_to_the_original_values(self):
        """Spec 4: regenerating the core-owned fields from an unchanged core reproduces them."""
        for name, truth in (("plain", phantom.Truth()), ("time", phantom.Truth(shape=(5, 6, 7, 3)))):
            with self.subTest(case=name):
                path = phantom.write_nifti(self.dir / f"{name}.nii.gz", truth, qcode=1, scode=2)
                image, _ = medvol.read(path)
                out = self.dir / f"{name}_regen.nii.gz"
                with mock.patch.object(NiftiPlugin, "_geometry_unchanged", return_value=False):
                    medvol.write(image, out)
                original, regenerated = nib.load(str(path)).header, nib.load(str(out)).header
                for key in ("srow_x", "srow_y", "srow_z", "qoffset_x", "qoffset_y", "qoffset_z", "qform_code",
                            "sform_code", "dim", "datatype", "bitpix", "xyzt_units"):
                    np.testing.assert_array_equal(regenerated[key], original[key], err_msg=key)
                np.testing.assert_allclose(regenerated["pixdim"][:5], original["pixdim"][:5], rtol=1e-6)
                for key in ("quatern_b", "quatern_c", "quatern_d"):
                    self.assertAlmostEqual(float(regenerated[key]), float(original[key]), places=6)

    def test_sheared_geometry_keeps_the_sform_exact_and_reports_the_qform(self):
        truth = phantom.sheared_truth()
        path = phantom.write_nifti(self.dir / "a.nii.gz", truth, qform=phantom.Truth().affine_ras)
        image, _ = medvol.read(path)
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        out = self.dir / "regen.nii.gz"
        with mock.patch.object(NiftiPlugin, "_geometry_unchanged", return_value=False):
            report = medvol.write(image, out)
        self.assertTrue(any("closest rigid" in w for w in report.warnings))
        again, _ = medvol.read(out)
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_changed_data_drops_display_range(self):
        path = phantom.write_nifti(self.dir / "a.nii.gz", extra={"cal_min": 1.0, "cal_max": 50.0})
        image, _ = medvol.read(path)
        image.array[0, 0, 0] += 1
        report = medvol.write(image, self.dir / "b.nii.gz")
        self.assertEqual(float(nib.load(str(self.dir / "b.nii.gz")).header["cal_max"]), 0.0)
        self.assertTrue(any("cal_max dropped" in w for w in report.warnings))


class TestFromOtherSources(NiftiCase):
    def test_mlarray_without_source_writes_correct_geometry(self):
        truth = phantom.Truth()
        nifti, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", truth))
        image = MLArray(nifti.array, axes=["z", "y", "x"], spaces=[nifti.space().to_dict()])
        out = self.dir / "b.nii.gz"
        medvol.write(medvol.Image.from_mlarray(image), out)
        header = nib.load(str(out)).header
        self.assertEqual((int(header["qform_code"]), int(header["sform_code"])), (1, 1))
        again, _ = medvol.read(out)
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_labels_and_user_metadata_are_reported_as_dropped(self):
        image = MLArray(np.zeros((2, 3, 4), dtype=np.uint8), spacing=(1, 1, 1), space="patient",
                        values={"kind": "label", "labels": {"1": {"name": "liver"}}}, user={"a": 1})
        report = medvol.write(medvol.Image.from_mlarray(image), self.dir / "seg.nii.gz")
        self.assertEqual(sorted(d.split(":")[0] for d in report.dropped), ["user", "values.labels"])


if __name__ == "__main__":
    unittest.main()
