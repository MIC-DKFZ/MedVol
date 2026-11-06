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
        with self.assertRaises(NotImplementedError):
            medvol.read(directory, dicom_tilt="resample")

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
        with self.assertRaisesRegex(GeometryError, "rescale"):
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


if __name__ == "__main__":
    unittest.main()
