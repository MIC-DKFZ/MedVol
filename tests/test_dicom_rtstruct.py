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
from medvol.plugins import dicom_rtstruct
from medvol.plugins.dicom import DicomPlugin
from medvol.policy import Policy
from medvol.report import LossError

warnings.filterwarnings("ignore", module="pydicom")

SHAPE = (20, 16, 5)          # (i columns, j rows, k slices)

#: polygons in (i, j) index coordinates of the truth; no voxel centre lies on an edge
BOX = np.array([[2.5, 3.5], [8.5, 3.5], [8.5, 9.5], [2.5, 9.5]])
TRIANGLE = np.array([[11.3, 1.2], [18.7, 2.9], [14.1, 14.6]])
HOLE = np.array([[13.5, 5.5], [15.5, 5.5], [15.5, 7.5], [13.5, 7.5]])
ISLAND = np.array([[14.7, 6.7], [15.3, 6.7], [15.3, 7.3], [14.7, 7.3]])     # inside the hole: centre (15, 7)


def _truth(sheared=False):
    affine = phantom.sheared_truth().affine_lps if sheared else phantom.Truth().affine_lps
    return phantom.Truth(shape=SHAPE, affine_lps=affine, dtype="int16")


# ---- expected masks, computed independently of MedVol (analytic, in the truth's index space) ----

def _centres():
    return np.meshgrid(np.arange(SHAPE[0], dtype=float), np.arange(SHAPE[1], dtype=float), indexing="ij")   # (i, j)


def _inside_rectangle(polygon):
    i, j = _centres()
    (i0, j0), (i1, j1) = polygon.min(axis=0), polygon.max(axis=0)
    return (i > i0) & (i < i1) & (j > j0) & (j < j1)


def _inside_triangle(polygon):
    i, j = _centres()
    signs = []
    for n in range(3):
        (ai, aj), (bi, bj) = polygon[n], polygon[(n + 1) % 3]
        cross = (bi - ai) * (j - aj) - (bj - aj) * (i - ai)
        length = np.hypot(bi - ai, bj - aj)
        assert np.min(np.abs(cross) / length) > 1e-3, "a voxel centre lies on a triangle edge"
        signs.append(cross > 0)
    return (signs[0] == signs[1]) & (signs[1] == signs[2])


def expected_masks():
    """ROI number -> boolean mask in the truth's file order (i, j, k)."""
    box = np.zeros(SHAPE, dtype=bool)
    box[:, :, 1] = box[:, :, 2] = _inside_rectangle(BOX)
    shape5 = np.zeros(SHAPE, dtype=bool)
    shape5[:, :, 3] = _inside_triangle(TRIANGLE) ^ _inside_rectangle(HOLE) ^ _inside_rectangle(ISLAND)
    shape5[:, :, 4] = _inside_rectangle(BOX)
    assert shape5[15, 7, 3] and not shape5[14, 6, 3] and shape5[14, 4, 3]       # island in, hole out, triangle in
    return {2: box, 5: shape5}


ROIS = [
    {"number": 2, "name": "box", "color": (255, 0, 0), "code": ("SCT", "10200004", "Liver"),
     "contours": [(1, BOX), (2, BOX[::-1])]},                                  # either winding
    {"number": 5, "name": "triangle with hole", "color": (0, 128, 255), "code": None,
     "contours": [(3, TRIANGLE), (3, HOLE), (3, ISLAND), (4, BOX)]},
]


class RtstructCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, sheared=False, rois=ROIS, **kwargs):
        truth = _truth(sheared)
        series = self.dir / "ct"
        phantom.write_dicom_series(series, truth)
        path = self.dir / "rs.dcm"
        phantom_dicom.write_rtstruct(path, series, truth, rois, **kwargs)
        return truth, series, path

    def truth_index(self, series, truth):
        """The truth's (i, j, k) of every voxel of the series image (values encode it)."""
        ct, _ = medvol.read(series)
        return ct, np.unravel_index(ct.array.astype(np.int64), truth.shape, order="F")


class TestRasterisation(RtstructCase):
    def test_rois_follow_the_centre_inside_even_odd_rule(self):
        for sheared in (False, True):                       # oblique grid, and a gantry-tilted (sheared) one
            with self.subTest(tilted=sheared):
                truth, series, path = self.make(sheared)
                images, report = dicom_rtstruct.read_rtstruct(path, series, Policy())
                self.assertEqual(len(images), 1)
                image = images[0]
                ct, (i, j, k) = self.truth_index(series, truth)
                values = image.meta.values
                self.assertEqual((values.kind, values.label_layout), ("label", "index"))
                self.assertEqual(image.array.shape, ct.array.shape)
                self.assertEqual(image.array.dtype, np.uint8)
                expected = np.zeros(ct.array.shape, dtype=np.uint8)
                for number, mask in expected_masks().items():
                    expected[mask[i, j, k]] = number
                np.testing.assert_array_equal(image.array, expected)
                np.testing.assert_array_equal(image.space().matrix(), ct.space().matrix())
                self.assertEqual(dict(image.space().reference), dict(ct.space().reference))
                labels = {key: dict(value) for key, value in values.labels.items()}
                self.assertEqual(sorted(labels), ["2", "5"])
                self.assertEqual(labels["2"]["name"], "box")
                self.assertEqual(list(labels["2"]["color"]), [255, 0, 0])
                self.assertEqual(dict(labels["2"]["code"]), {"scheme": "SCT", "value": "10200004", "meaning": "Liver"})
                self.assertNotIn("code", labels["5"])
                self.assertIn("even-odd", report.choices["rtstruct_rasterisation"])
                self.assertEqual(report.dropped, [])
                header = image.meta.source[0].header
                self.assertEqual(header["shared"]["Modality"], "RTSTRUCT")
                self.assertIn("ROIContourSequence", header["shared"])
                self.assertEqual(list(header["file_names"]), ["rs.dcm"])

    def test_overlapping_rois_are_channels(self):
        rois = [dict(ROIS[0]), {"number": 7, "name": "overlap", "color": None, "code": None,
                                "contours": [(2, BOX + [3.0, 2.0])]}]
        truth, series, path = self.make(rois=rois)
        images, report = dicom_rtstruct.read_rtstruct(path, series, Policy())
        image = images[0]
        self.assertEqual(image.meta.values.label_layout, "channel")
        self.assertEqual([(a.name, a.type) for a in image.meta.axes][0], ("c", "channel"))
        ct, (i, j, k) = self.truth_index(series, truth)
        box = expected_masks()[2]
        moved = np.zeros(SHAPE, dtype=bool)
        moved[:, :, 2] = _inside_rectangle(BOX + [3.0, 2.0])
        np.testing.assert_array_equal(image.array[0], box[i, j, k])
        np.testing.assert_array_equal(image.array[1], moved[i, j, k])
        self.assertEqual(image.meta.values.labels["1"]["name"], "overlap")
        self.assertTrue(any("overlap" in w for w in report.warnings))

    def test_directory_with_series_and_rtstruct(self):
        truth, series, path = self.make()
        path.rename(series / "rs.dcm")
        images, _ = medvol.read_all(series)
        self.assertEqual(sorted(image.meta.values.kind for image in images), ["label", "measurement"])
        label = next(image for image in images if image.meta.values.kind == "label")
        alone, _ = DicomPlugin().read(series / "rs.dcm", Policy(), reference=series)
        via_api, _ = medvol.read(series / "rs.dcm", reference=series)
        np.testing.assert_array_equal(via_api.array, alone[0].array)
        np.testing.assert_array_equal(label.array, alone[0].array)
        with self.assertRaisesRegex(ValueError, "reference"):
            medvol.read(series / "rs.dcm")                   # the referenced series is not given

    def test_other_frame_of_reference_is_refused(self):
        _, series, path = self.make(frame_of_reference=pydicom.uid.generate_uid())
        with self.assertRaisesRegex(GeometryError, "Frame of Reference"):
            dicom_rtstruct.read_rtstruct(path, series, Policy())

    def test_off_plane_open_and_outside_contours_are_reported(self):
        rois = [{"number": 1, "name": "odd", "color": None, "code": None,
                 "contours": [(1, BOX), (2.5, BOX), (3, BOX, "OPEN_PLANAR"), (9, BOX)]}]
        truth, series, path = self.make(rois=rois)
        images, report = dicom_rtstruct.read_rtstruct(path, series, Policy())
        self.assertEqual(len(report.dropped), 2)
        self.assertTrue(any("off every slice plane" in d for d in report.dropped))
        self.assertTrue(any("beyond the first or last slice" in d for d in report.dropped))
        self.assertTrue(any("OPEN_PLANAR" in w for w in report.warnings))
        ct, (i, j, k) = self.truth_index(series, truth)
        np.testing.assert_array_equal(images[0].array > 0, (k == 1) & _inside_rectangle(BOX)[i, j])
        with self.assertRaises(LossError):
            dicom_rtstruct.read_rtstruct(path, series, Policy(on_loss="error"))


class TestEvenOdd(unittest.TestCase):
    def test_holes_and_islands_by_even_odd(self):
        outer = np.array([[0.5, 0.5], [0.5, 6.5], [6.5, 6.5], [6.5, 0.5]])     # (row, column)
        hole = np.array([[1.5, 1.5], [1.5, 5.5], [5.5, 5.5], [5.5, 1.5]])
        island = np.array([[2.5, 2.5], [2.5, 4.5], [4.5, 4.5], [4.5, 2.5]])
        mask = dicom_rtstruct.even_odd([outer, hole, island], 8, 8)
        expected = np.zeros((8, 8), dtype=bool)
        expected[1:7, 1:7] = True
        expected[2:6, 2:6] = False
        expected[3:5, 3:5] = True
        np.testing.assert_array_equal(mask, expected)

    def test_centres_on_edges_follow_the_half_open_test(self):
        square = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])     # edges through centres
        mask = dicom_rtstruct.even_odd([square], 5, 5)
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:3, 1:3] = True            # left edge and lower row bound inside, right edge and upper bound outside
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
