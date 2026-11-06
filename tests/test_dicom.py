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


if __name__ == "__main__":
    unittest.main()
