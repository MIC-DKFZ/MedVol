import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pydicom

import medvol
import phantom
from medvol.geometry import GeometryError

warnings.filterwarnings("ignore", module="pydicom")


def _truth(sheared=False, shape=phantom.SHAPE):
    truth = phantom.sheared_truth() if sheared else phantom.Truth(shape=shape)
    truth.dtype = "int16"
    return truth


class DicomCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def series(self, name="series", truth=None, **kwargs):
        directory = self.dir / name
        phantom.write_dicom_series(directory, truth or _truth(), **kwargs)
        return directory


class TestSingleFrameSeries(DicomCase):
    def test_world_positions_of_regular_and_tilted_series(self):
        for sheared in (False, True):
            with self.subTest(tilted=sheared):
                truth = _truth(sheared)
                image, report = medvol.read(self.series(f"s{sheared}", truth))
                self.assertLess(phantom.world_error(image, truth), 1e-4)
                self.assertEqual(image.array.shape, (7, 6, 5))
                self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
                self.assertEqual(bool(report.choices.get("dicom_tilt")), sheared)

    def test_tilt_policy(self):
        directory = self.series(truth=_truth(sheared=True))
        with self.assertRaises(GeometryError):
            medvol.read(directory, dicom_tilt="error")
        image, report = medvol.read(directory, dicom_tilt="resample")     # TestResample checks the values
        self.assertTrue(report.choices["dicom_tilt"].startswith("resample"))

    def test_ras_frame_and_frame_of_reference(self):
        truth = _truth()
        directory = self.series(truth=truth)
        image, _ = medvol.read(directory, world_frame="ras")
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        uid = pydicom.dcmread(next(directory.iterdir())).FrameOfReferenceUID
        self.assertEqual(dict(image.space().reference), {"scheme": "dicom_frame_of_reference", "id": uid})

    def test_values_follow_rescale_and_padding(self):
        directory = self.series(extra={"RescaleSlope": 2, "RescaleIntercept": -1024, "PixelPaddingValue": -2000})
        image, _ = medvol.read(directory)
        values = image.meta.values
        self.assertEqual((values.scale, values.offset, values.fill, values.unit), (2.0, -1024.0, -2000, "hounsfield_unit"))
        self.assertEqual(image.array.dtype, np.int16)

    def test_header_holds_shared_and_per_file_fields(self):
        image, _ = medvol.read(self.series())
        header = image.meta.source[0].header
        self.assertEqual(header["shared"]["Modality"], "CT")
        self.assertEqual(header["shared"]["(0029,0010)"], {"vr": "LO", "value": "SYNTH PRIVATE"})
        self.assertEqual(len(header["per_file"]), 7)
        self.assertIn("(0029,1010)", header["per_file"][0])
        self.assertEqual(header["per_file"][0]["(0029,1010)"]["vr"], "OB")
        self.assertIsInstance(header["per_file"][0]["(0029,1010)"]["value"], bytes)
        self.assertNotIn("PixelData", header["shared"])
        positions = [f["ImagePositionPatient"] for f in header["per_file"]]
        normal = np.cross(*np.reshape(header["shared"]["ImageOrientationPatient"], (2, 3)))
        self.assertTrue(np.all(np.diff(np.array(positions) @ normal) > 0))   # slice order, not file order
        self.assertEqual(len(header["file_names"]), 7)


class TestAssemblyPolicies(DicomCase):
    def test_localizer_in_the_series_is_split_kept_or_refused(self):
        main = self.series()
        uid = pydicom.dcmread(next(main.iterdir())).SeriesInstanceUID
        localizer = phantom.Truth(shape=(5, 6, 1), affine_lps=np.diag([0.7, 0.7, 1.0, 1.0]), dtype="int16")
        phantom.write_dicom_series(main / "loc", localizer, extra={"SeriesInstanceUID": uid})
        images, report = medvol.read_all(main)
        self.assertEqual(sorted(i.array.shape for i in images), [(1, 6, 5), (7, 6, 5)])
        with self.assertRaises(ValueError):
            medvol.read(main)
        image, _ = medvol.read(main, dicom_stacks="largest")
        self.assertLess(phantom.world_error(image, _truth()), 1e-4)
        with self.assertRaises(GeometryError):
            medvol.read(main, dicom_stacks="error")

    def test_irregular_spacing_is_refused_or_split(self):
        truth = _truth()
        directory = self.series(truth=truth, order=[0, 1, 2, 4, 5, 6])    # slice 3 missing
        with self.assertRaises(GeometryError):
            medvol.read(directory)
        images, report = medvol.read_all(directory, dicom_irregular_spacing="split")
        self.assertEqual(sorted(i.array.shape[0] for i in images), [3, 3])
        for image in images:
            self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_several_series_are_separate_images(self):
        self.series("a")
        self.series("b")
        images, report = medvol.read_all(self.dir)
        self.assertEqual(len(images), 2)
        self.assertIn("2 series", report.warnings[0])

    def test_duplicate_positions_and_varying_rescale_are_refused(self):
        directory = self.series()
        files = sorted(directory.iterdir())
        ds = pydicom.dcmread(files[0])
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.save_as(directory / "duplicate.dcm", enforce_file_format=True)
        with self.assertRaisesRegex(GeometryError, "share one slice position"):
            medvol.read(directory)
        (directory / "duplicate.dcm").unlink()
        ds = pydicom.dcmread(files[1])
        ds.RescaleSlope = 3
        ds.save_as(files[1], enforce_file_format=True)
        with self.assertRaisesRegex(GeometryError, "rescale.*dicom_rescale='error'"):
            medvol.read(directory)

    def test_single_slice_uses_the_slice_thickness(self):
        truth = phantom.Truth(shape=(5, 6, 1), dtype="int16")
        image, report = medvol.read(self.series(truth=truth))
        self.assertEqual(image.array.shape, (1, 6, 5))
        self.assertAlmostEqual(image.space().spacing[0], 1.5, places=9)
        self.assertIn("single slice", report.warnings[0])


def _load(directory):
    return {p.relative_to(directory).as_posix(): pydicom.dcmread(p) for p in sorted(Path(directory).rglob("*.dcm"))}


def _differences(a, b):
    diffs = [e.keyword or str(e.tag) for e in a if e.tag != 0x7FE00010 and (e.tag not in b or b[e.tag].value != e.value or b[e.tag].VR != e.VR)]
    diffs += [e.keyword or str(e.tag) for e in b if e.tag not in a]
    diffs += ["file_meta." + (e.keyword or str(e.tag)) for e in a.file_meta if e.tag not in b.file_meta or b.file_meta[e.tag].value != e.value]
    return diffs


class TestWrite(DicomCase):
    def round_trip(self, directory, **options):
        image, _ = medvol.read(directory, **options)
        medvol.write(image, self.dir / "x.mla")
        again, _ = medvol.read(self.dir / "x.mla")
        out = self.dir / "out"
        report = medvol.write(again, out, format="dicom")
        return out, report

    def test_series_round_trip_reemits_every_file_exactly(self):
        for name, kwargs in (("regular", {}), ("tilted", {"truth": _truth(sheared=True)}),
                             ("rescaled", {"extra": {"RescaleSlope": 2, "RescaleIntercept": -1024}})):
            with self.subTest(series=name):
                directory = self.series(name, **kwargs)
                out, report = self.round_trip(directory)
                original, written = _load(directory), _load(out)
                self.assertEqual(sorted(original), sorted(written))
                for file in original:
                    self.assertEqual(_differences(original[file], written[file]), [], file)
                    np.testing.assert_array_equal(original[file].pixel_array, written[file].pixel_array)
                self.assertEqual(report.warnings, [])
                shutil.rmtree(out)

    def test_rle_lossless_is_kept(self):
        directory = self.series()
        for path in directory.iterdir():
            ds = pydicom.dcmread(path)
            ds.compress(pydicom.uid.RLELossless)
            ds.save_as(path, enforce_file_format=True)
        out, _ = self.round_trip(directory)
        for ds in _load(out).values():
            self.assertEqual(ds.file_meta.TransferSyntaxUID, pydicom.uid.RLELossless)

    def test_changed_geometry_is_regenerated(self):
        truth = _truth()
        image, _ = medvol.read(self.series(truth=truth))
        moved = image.space().matrix()
        moved[:3, -1] += [1.0, 2.0, 3.0]
        image.meta.spaces = [dict(image.space().to_dict(), from_array={"array_axes": ["z", "y", "x"], "matrix": moved.tolist()})]
        out = self.dir / "out"
        report = medvol.write(image, out, format="dicom")
        self.assertEqual(report.choices["geometry"], "regenerated from the core")
        again, _ = medvol.read(out)
        np.testing.assert_allclose(again.space().matrix(), moved, atol=1e-6)

    def test_changed_data_gets_new_instance_uids(self):
        directory = self.series()
        image, _ = medvol.read(directory)
        image.array[0, 0, 0] += 1
        out = self.dir / "out"
        report = medvol.write(image, out, format="dicom")
        self.assertTrue(any("new SOP Instance UIDs" in w for w in report.warnings))
        original = {ds.InstanceNumber: ds.SOPInstanceUID for ds in _load(directory).values()}
        written = {ds.InstanceNumber: ds.SOPInstanceUID for ds in _load(out).values()}
        self.assertTrue(all(original[n] != written[n] for n in original))

    def test_only_dicom_sources_are_written_as_dicom(self):
        nifti, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz"))
        with self.assertRaises(NotImplementedError):
            medvol.write(nifti, self.dir / "out", format="dicom")


def _set_rescale(directory, rescale):
    """Give every file of a series its own (slope, intercept): ``rescale(slice k) -> pair``."""
    for path in sorted(Path(directory).iterdir()):
        ds = pydicom.dcmread(path)
        k = int(round(float(np.dot(np.array(ds.ImagePositionPatient, dtype=float) - phantom.ORIGIN_LPS,
                                   phantom.DIRECTION_LPS[:, 2])) / phantom.SPACING[2]))
        ds.RescaleSlope, ds.RescaleIntercept = rescale(k)
        ds.save_as(path, enforce_file_format=True)


def _stored_by_slice(directory):
    """Stored pixels and (slope, intercept) of every file, keyed by SOP Instance UID."""
    return {ds.SOPInstanceUID: ds for ds in _load(directory).values()}


class TestRescaleApply(DicomCase):
    def check_physical(self, image, directory):
        """The array is slope * stored + intercept with each slice's own file."""
        header = image.meta.source[0].header
        datasets = _stored_by_slice(directory)
        for z, fields in enumerate(header["per_file"]):
            merged = {**header["shared"], **fields}
            ds = datasets[merged["SOPInstanceUID"]]
            physical = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            np.testing.assert_array_equal(image.array[z].astype(np.float64), physical)

    def test_apply_stores_physical_values_and_writes_back_exactly(self):
        for name, rescale, dtype in (("exact32", lambda k: (1 + k % 3, -1024 + 10 * k), np.float32),
                                     ("needs64", lambda k: ((0.1, 0.3, 0.7)[k % 3], -3.7), np.float64)):
            with self.subTest(case=name):
                directory = self.series(name)
                _set_rescale(directory, rescale)
                with self.assertRaisesRegex(GeometryError, "dicom_rescale"):
                    medvol.read(directory)
                image, report = medvol.read(directory, dicom_rescale="apply")
                self.assertEqual(image.array.dtype, dtype)
                values = image.meta.values
                self.assertEqual((values.scale, values.offset, values.unit), (None, None, "hounsfield_unit"))
                self.assertTrue(report.choices["dicom_rescale"].startswith("apply"))
                self.check_physical(image, directory)
                medvol.write(image, self.dir / f"{name}.mla")
                again, _ = medvol.read(self.dir / f"{name}.mla")
                out = self.dir / f"out_{name}"
                report = medvol.write(again, out, format="dicom")
                self.assertEqual(report.warnings, [])
                original, written = _load(directory), _load(out)
                for file in original:
                    self.assertEqual(_differences(original[file], written[file]), [], file)
                    np.testing.assert_array_equal(original[file].pixel_array, written[file].pixel_array)

    def test_changed_physical_values_are_written_if_exact_else_refused(self):
        from medvol.plugins.dicom_rescale import RescaleError

        directory = self.series()
        _set_rescale(directory, lambda k: (1 + k % 3, -1024.0))
        image, _ = medvol.read(directory, dicom_rescale="apply")
        header = image.meta.source[0].header
        slope = float({**header["shared"], **header["per_file"][0]}["RescaleSlope"])
        image.array[0, 0, 0] += slope                      # one stored step: still exact
        report = medvol.write(image, self.dir / "ok", format="dicom")
        self.assertTrue(any("new SOP Instance UIDs" in w for w in report.warnings))
        again, _ = medvol.read(self.dir / "ok", dicom_rescale="apply")
        np.testing.assert_array_equal(again.array, image.array)
        image.array[0, 0, 0] += 0.25 if slope > 1 else 0.5   # between two stored steps
        with self.assertRaisesRegex(RescaleError, "refused"):
            medvol.write(image, self.dir / "bad", format="dicom")


class TestResample(DicomCase):
    def check(self, image, truth, exact_coverage=True):
        """Values equal the analytic phantom; NaN exactly outside the source (``exact_coverage``).

        With a gap in a tilted stack, a voxel needs both bracketing slices, so the covered region
        is smaller than the analytic box; then only the covered voxels are compared.
        """
        from medvol.geometry import has_shear, lps_matrix

        linear = lps_matrix(image.space())[:3, :3]
        self.assertFalse(has_shear(linear))
        expected = phantom.expected_linear(image, truth)
        if exact_coverage:
            np.testing.assert_array_equal(np.isnan(image.array), np.isnan(expected))
        covered = ~np.isnan(image.array)
        self.assertFalse(np.any(np.isnan(expected[covered])))
        self.assertGreater(np.count_nonzero(covered), image.array.size // 3)
        np.testing.assert_allclose(image.array[covered], expected[covered], rtol=0, atol=1e-4)

    def test_tilted_series_resampled_onto_a_grid_without_shear(self):
        truth = _truth(sheared=True)
        directory = self.series(truth=truth)
        image, report = medvol.read(directory, dicom_tilt="resample")
        self.assertEqual(image.array.shape, (7, 6, 5))
        self.assertEqual(image.array.dtype, np.float32)
        self.check(image, truth)
        self.assertTrue(np.isnan(image.meta.values.fill))
        self.assertTrue(np.any(np.isnan(image.array)) and np.any(np.isfinite(image.array)))
        space = image.space()
        normal = np.cross(*np.reshape(image.meta.source[0].header["shared"]["ImageOrientationPatient"], (2, 3)))
        step = lps_column(space, "z")
        self.assertAlmostEqual(abs(float(step @ normal)), np.linalg.norm(step), places=9)      # along the normal
        self.assertAlmostEqual(np.linalg.norm(step), 2.0, places=9)
        text = " ".join(report.warnings)
        self.assertIn("trilinear", text)
        self.assertIn("relabelling the source slices onto this grid without resampling would misplace voxels by up to 3.6 mm", text)
        fp = image.meta.source[0].fingerprint
        from mlarray import fingerprint

        self.assertNotEqual(fp["geometry"], fingerprint.geometry_hash(image.meta))
        self.assertNotEqual(fp["data"], fingerprint.data_hash(image.array))
        medvol.write(image, self.dir / "x.mla")                                  # NaN fill survives .mla
        again, _ = medvol.read(self.dir / "x.mla")
        np.testing.assert_array_equal(again.array, image.array)

    def test_irregular_series_resampled_at_the_mean_spacing(self):
        truth = _truth()
        directory = self.series(truth=truth, order=[0, 1, 2, 4, 5, 6])          # slice 3 missing
        image, report = medvol.read(directory, dicom_irregular_spacing="resample")
        self.assertEqual(image.array.shape, (6, 6, 5))
        self.assertAlmostEqual(np.linalg.norm(lps_column(image.space(), "z")), 12.0 / 5, places=9)
        self.check(image, truth)
        self.assertFalse(np.any(np.isnan(image.array)))
        self.assertIsNone(image.meta.values.fill)
        self.assertTrue(report.choices["dicom_irregular_spacing"].startswith("resample"))

    def test_irregular_and_tilted_needs_both_resample_policies(self):
        truth = _truth(sheared=True)
        directory = self.series(truth=truth, order=[0, 1, 2, 4, 5, 6])
        with self.assertRaisesRegex(GeometryError, "dicom_tilt='resample'"):
            medvol.read(directory, dicom_irregular_spacing="resample")
        image, report = medvol.read(directory, dicom_irregular_spacing="resample", dicom_tilt="resample")
        self.check(image, truth, exact_coverage=False)
        self.assertIn("dicom_tilt", report.choices)

    def test_writing_a_resampled_image_as_dicom_is_refused(self):
        from medvol.plugins.dicom_rescale import RescaleError

        directory = self.series(truth=_truth(sheared=True))
        image, _ = medvol.read(directory, dicom_tilt="resample")
        with self.assertRaisesRegex(RescaleError, "cannot write"):        # interpolated values are not stored integers
            medvol.write(image, self.dir / "out", format="dicom")
        from medvol.image import Image

        cropped = Image(np.ascontiguousarray(image.array[:5]), image.meta.copy())
        with self.assertRaisesRegex(GeometryError, "slice count changed"):
            medvol.write(cropped, self.dir / "out2", format="dicom")

    def test_integer_valued_resampling_is_written_with_regenerated_geometry(self):
        truth = _truth()
        directory = self.series(truth=truth, order=[0, 1, 2, 3, 4, 6])           # 0..4 at 2 mm, then 4 mm: mean 2.4 mm
        image, _ = medvol.read(directory, dicom_irregular_spacing="resample")
        image.array[...] = np.round(image.array)                                 # make them stored integers again
        out = self.dir / "out"
        report = medvol.write(image, out, format="dicom")
        self.assertEqual(report.choices["geometry"], "regenerated from the core")
        self.assertTrue(any("new SOP Instance UIDs" in w for w in report.warnings))
        again, _ = medvol.read(out)
        np.testing.assert_allclose(again.space().matrix(), image.space().matrix(), atol=1e-6)
        np.testing.assert_array_equal(again.array, image.array.astype(np.int16))


def lps_column(space, axis):
    from medvol.geometry import lps_matrix

    return lps_matrix(space)[:3, list(space.from_array.array_axes).index(axis)]


if __name__ == "__main__":
    unittest.main()
