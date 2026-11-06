import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pydicom

import medvol
import phantom
import phantom_dicom
from medvol.geometry import GeometryError

warnings.filterwarnings("ignore", module="pydicom")

SHAPE_4D = (5, 6, 7, 3)


def _truth(shape=phantom.SHAPE, sheared=False):
    truth = phantom.sheared_truth() if sheared else phantom.Truth(shape=shape)
    truth.dtype = "int16"
    return truth


def _time_index(image, truth):
    """The time index each voxel's value encodes (4-D phantom)."""
    return np.unravel_index(image.array.astype(np.int64), truth.shape, order="F")[3]


def _differences(a, b):
    diffs = [e.keyword or str(e.tag) for e in a if e.tag != 0x7FE00010 and (e.tag not in b or b[e.tag].value != e.value or b[e.tag].VR != e.VR)]
    diffs += [e.keyword or str(e.tag) for e in b if e.tag not in a]
    diffs += ["file_meta." + (e.keyword or str(e.tag)) for e in a.file_meta if e.tag not in b.file_meta or b.file_meta[e.tag].value != e.value]
    return diffs


class EnhancedCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def enhanced(self, name="enhanced.dcm", truth=None, **kwargs):
        path = self.dir / name
        phantom_dicom.write_enhanced(path, truth or _truth(), **kwargs)
        return path


class TestEnhanced3D(EnhancedCase):
    def test_scrambled_frames_are_placed_by_position(self):
        for per_frame in (False, True):            # Canon-like shared vs Philips/Siemens-like per-frame geometry
            with self.subTest(per_frame_geometry=per_frame):
                truth = _truth()
                path = self.enhanced(f"ct{per_frame}.dcm", truth, per_frame_geometry=per_frame)
                frame_order = [int(f.FrameContentSequence[0].InStackPositionNumber)
                               for f in pydicom.dcmread(path).PerFrameFunctionalGroupsSequence]
                self.assertNotEqual(frame_order, sorted(frame_order))           # really scrambled
                image, report = medvol.read(path)
                self.assertEqual(image.array.shape, (7, 6, 5))
                self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
                self.assertLess(phantom.world_error(image, truth), 1e-4)
                self.assertEqual(image.array.dtype, np.int16)

    def test_highdicom_legacy_converted_enhanced_ct(self):
        import highdicom as hd

        truth = _truth()
        series = self.dir / "series"
        phantom_dicom.write_ct_series(series, truth)
        legacy = [pydicom.dcmread(p) for p in sorted(series.iterdir())]
        converted = hd.legacy.LegacyConvertedEnhancedCTImage(
            legacy_datasets=legacy, series_instance_uid=hd.UID(), series_number=10, sop_instance_uid=hd.UID(), instance_number=1)
        converted.save_as(self.dir / "legacy.dcm")
        image, _ = medvol.read(self.dir / "legacy.dcm")
        self.assertEqual(image.array.shape, (7, 6, 5))
        self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_values_and_frame_of_reference(self):
        path = self.enhanced(rescale=(2.0, -1024.0))
        image, _ = medvol.read(path, world_frame="ras")
        values = image.meta.values
        self.assertEqual((values.kind, values.scale, values.offset, values.unit), ("measurement", 2.0, -1024.0, "hounsfield_unit"))
        self.assertLess(phantom.world_error(image, _truth()), 1e-4)
        uid = pydicom.dcmread(path).FrameOfReferenceUID
        self.assertEqual(dict(image.space().reference), {"scheme": "dicom_frame_of_reference", "id": uid})

    def test_header_keeps_shared_groups_and_per_frame_groups_in_array_order(self):
        path = self.enhanced()
        original = pydicom.dcmread(path)
        image, _ = medvol.read(path)
        header = image.meta.source[0].header
        self.assertIn("SharedFunctionalGroupsSequence", header["shared"])
        self.assertNotIn("PerFrameFunctionalGroupsSequence", header["shared"])
        self.assertEqual(header["shared"]["(0029,0010)"], {"vr": "LO", "value": "SYNTH PRIVATE"})
        self.assertEqual(len(header["frames"]), 7)
        normal = np.cross(*np.reshape(original.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient, (2, 3)).astype(float))
        positions = [entry["functional_groups"]["PlanePositionSequence"][0]["ImagePositionPatient"] for entry in header["frames"]]
        self.assertTrue(np.all(np.diff(np.array(positions) @ normal) > 0))
        for slot, entry in enumerate(header["frames"]):          # frame_index points at the original frame
            self.assertEqual(entry["file"], 0)
            np.testing.assert_array_equal(original.pixel_array[entry["frame_index"]], image.array[slot])

    def test_tilt_policy(self):
        truth = _truth(sheared=True)
        path = self.enhanced(truth=truth)
        image, report = medvol.read(path)
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assertEqual(report.choices["dicom_tilt"], "exact")
        with self.assertRaises(GeometryError):
            medvol.read(path, dicom_tilt="error")
        with self.assertRaises(NotImplementedError):
            medvol.read(path, dicom_tilt="resample")

    def test_irregular_spacing_is_refused_or_split(self):
        truth = _truth()
        path = self.enhanced(truth=truth, order=[5, 0, 2, 6, 1, 4])            # slice 3 missing
        with self.assertRaises(GeometryError):
            medvol.read(path)
        images, _ = medvol.read_all(path, dicom_irregular_spacing="split")
        self.assertEqual(sorted(i.array.shape[0] for i in images), [3, 3])
        for image in images:
            self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_frames_at_one_position_without_a_dimension_are_refused(self):
        path = self.enhanced("4d.dcm", _truth(SHAPE_4D), mr=True)
        ds = pydicom.dcmread(path)
        del ds.DimensionIndexSequence
        for item in ds.PerFrameFunctionalGroupsSequence:
            del item.FrameContentSequence[0].DimensionIndexValues
            del item.FrameContentSequence[0].TemporalPositionIndex
        ds.save_as(path, enforce_file_format=True)
        with self.assertRaisesRegex(GeometryError, "share one slice position"):
            medvol.read(path)


class TestEnhanced4D(EnhancedCase):
    def check_4d(self, image, truth):
        self.assertEqual(image.array.shape, (3, 7, 6, 5))
        self.assertEqual([(a.name, a.type) for a in image.meta.axes], [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")])
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        np.testing.assert_array_equal(_time_index(image, truth), np.broadcast_to(np.arange(3)[:, None, None, None], image.array.shape))

    def test_enhanced_mr_4d(self):
        truth = _truth(SHAPE_4D)
        image, report = medvol.read(self.enhanced("mr4d.dcm", truth, mr=True, per_frame_geometry=True))
        self.check_4d(image, truth)
        self.assertIn("time", report.choices["dicom_4d_axis"])
        self.assertEqual(list(image.meta.source[0].index_map["axis_order"]), [3, 2, 1, 0])
        self.assertEqual(len(image.meta.source[0].header["frames"]), 21)

    def test_one_file_per_time_point(self):
        truth = _truth(SHAPE_4D)
        directory = self.dir / "xa"
        phantom_dicom.write_enhanced(directory, truth, mr=True, per_frame_geometry=True, file_per_time_point=True)
        image, _ = medvol.read(directory)
        self.check_4d(image, truth)
        header = image.meta.source[0].header
        self.assertEqual(len(header["file_names"]), 3)
        self.assertEqual(len({entry["file"] for entry in header["frames"]}), 3)

    def test_other_dimension_gives_an_other_axis(self):
        truth = _truth(SHAPE_4D)
        path = self.enhanced("echo.dcm", truth, mr=True)
        ds = pydicom.dcmread(path)
        ds.DimensionIndexSequence[2].DimensionIndexPointer = 0x00189082        # Effective Echo Time
        for item in ds.PerFrameFunctionalGroupsSequence:
            del item.FrameContentSequence[0].TemporalPositionIndex
        ds.save_as(path, enforce_file_format=True)
        image, _ = medvol.read(path)
        self.assertEqual([(a.name, a.type) for a in image.meta.axes][0], ("u", "other"))
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        np.testing.assert_array_equal(_time_index(image, truth), np.broadcast_to(np.arange(3)[:, None, None, None], image.array.shape))

    def test_two_varying_dimensions_are_split_per_policy(self):
        truth = _truth(SHAPE_4D)
        path = self.enhanced("two.dcm", truth, mr=True)
        ds = pydicom.dcmread(path)
        echo = pydicom.dataset.Dataset()
        echo.DimensionIndexPointer, echo.FunctionalGroupPointer = 0x00189082, 0x00189114
        ds.DimensionIndexSequence.append(echo)
        for item in ds.PerFrameFunctionalGroupsSequence:          # echo 1 for time points 1-2, echo 2 for 3
            content = item.FrameContentSequence[0]
            content.DimensionIndexValues = list(content.DimensionIndexValues) + [1 if content.TemporalPositionIndex < 3 else 2]
        ds.save_as(path, enforce_file_format=True)
        images, report = medvol.read_all(path)
        self.assertEqual(sorted((i.array.shape for i in images), key=len), [(7, 6, 5), (2, 7, 6, 5)])
        for image in images:
            self.assertLess(phantom.world_error(image, truth), 1e-4)
        image, _ = medvol.read(path, dicom_stacks="largest")
        self.assertEqual(image.array.shape, (2, 7, 6, 5))
        with self.assertRaises(GeometryError):
            medvol.read(path, dicom_stacks="error")


class TestEnhancedWrite(EnhancedCase):
    def round_trip(self, source):
        image, _ = medvol.read(source)
        medvol.write(image, self.dir / "x.mla")
        again, _ = medvol.read(self.dir / "x.mla")
        out = self.dir / "out"
        report = medvol.write(again, out, format="dicom")
        return out, report

    def test_reemits_every_file_exactly(self):
        cases = {
            "ct": lambda: [self.enhanced("ct.dcm", rescale=(2.0, -1024.0))],
            "tilted": lambda: [self.enhanced("tilted.dcm", _truth(sheared=True))],
            "mr4d": lambda: [self.enhanced("mr4d.dcm", _truth(SHAPE_4D), mr=True, per_frame_geometry=True)],
        }
        for name, make in cases.items():
            with self.subTest(case=name):
                path = make()[0]
                out, report = self.round_trip(path)
                written = out / path.name
                a, b = pydicom.dcmread(path), pydicom.dcmread(written)
                self.assertEqual(_differences(a, b), [])
                np.testing.assert_array_equal(a.pixel_array, b.pixel_array)
                self.assertEqual(report.warnings, [])
                written.unlink()

    def test_multi_file_series_round_trip(self):
        directory = self.dir / "xa"
        phantom_dicom.write_enhanced(directory, _truth(SHAPE_4D), mr=True, per_frame_geometry=True, file_per_time_point=True)
        out, _ = self.round_trip(directory)
        for path in sorted(directory.iterdir()):
            a, b = pydicom.dcmread(path), pydicom.dcmread(out / path.name)
            self.assertEqual(_differences(a, b), [])
            np.testing.assert_array_equal(a.pixel_array, b.pixel_array)

    def test_changed_data_gets_new_instance_uid_and_changed_geometry_is_refused(self):
        path = self.enhanced()
        image, _ = medvol.read(path)
        image.array[0, 0, 0] += 1
        report = medvol.write(image, self.dir / "out", format="dicom")
        self.assertTrue(any("new SOP Instance UIDs" in w for w in report.warnings))
        written = pydicom.dcmread(self.dir / "out" / path.name)
        self.assertNotEqual(written.SOPInstanceUID, pydicom.dcmread(path).SOPInstanceUID)
        again, _ = medvol.read(self.dir / "out")
        np.testing.assert_array_equal(again.array, image.array)

        moved = image.space().matrix()
        moved[:3, -1] += 1.0
        image.meta.spaces = [dict(image.space().to_dict(), from_array={"array_axes": ["z", "y", "x"], "matrix": moved.tolist()})]
        with self.assertRaises(NotImplementedError):
            medvol.write(image, self.dir / "out2", format="dicom")


if __name__ == "__main__":
    unittest.main()
