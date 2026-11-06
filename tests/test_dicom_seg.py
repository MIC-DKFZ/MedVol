import random
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pydicom
from mlarray import Meta
from mlarray.metadata import thawed

import medvol
import phantom
import phantom_dicom
from medvol.geometry import GeometryError, lps_matrix
from medvol.plugins import dicom_seg

warnings.filterwarnings("ignore", module="pydicom")
warnings.filterwarnings("ignore", module="highdicom")


def _masks(truth, overlap=False):
    """Two segments in the series' file order (i, j, k); slices k = 0 and 3 stay empty."""
    first = np.zeros(truth.shape, dtype=bool)
    first[1:3, 2:5, 1:3] = True
    second = np.zeros(truth.shape, dtype=bool)
    second[0, 0, 5] = True
    second[3:5, 1:3, 4:6] = True
    if overlap:
        second[2, 3, 2] = True            # also in the first segment
    return {1: first, 2: second}


def _voxel_world(image, mask_of_array_index):
    """LPS world positions (N x 3) of the array voxels selected by a boolean array over the space axes."""
    space = image.space()
    matrix = lps_matrix(space)
    names = [axis.name for axis in image.meta.axes if axis.type == "space"]
    columns = [names.index(name) for name in space.from_array.array_axes]
    index = np.array(np.nonzero(mask_of_array_index))
    return (matrix @ np.vstack([index[columns], np.ones(index.shape[1])]))[:3].T


def _segment_error(test, image, truth, mask, selected):
    """Every selected voxel must be a truth voxel of ``mask`` and every mask voxel selected (exact in mm)."""
    world = _voxel_world(image, selected)
    inverse = np.linalg.inv(truth.affine_lps)
    ijk = (inverse @ np.vstack([world.T, np.ones(len(world))]))[:3]
    rounded = np.round(ijk).astype(int)
    back = (truth.affine_lps @ np.vstack([rounded, np.ones(rounded.shape[1])]))[:3].T
    error = float(np.max(np.linalg.norm(back - world, axis=1))) if len(world) else 0.0
    test.assertLess(error, 1e-4)
    test.assertEqual(sorted(map(tuple, rounded.T.tolist())), sorted(map(tuple, np.argwhere(mask).tolist())))


class SegCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.truth = phantom.Truth(dtype="int16")
        self.series = self.dir / "ct"
        phantom_dicom.write_ct_series(self.series, self.truth)

    def tearDown(self):
        self._tmp.cleanup()

    def seg(self, name="seg.dcm", masks=None, **kwargs):
        return phantom_dicom.write_seg(self.dir / name, self.series, masks or _masks(self.truth), **kwargs)

    def check_labels(self, labels, keys):
        for key, number in keys.items():
            expected = phantom_dicom.SEGMENTS[number]
            scheme, value, meaning = expected["code"]
            self.assertEqual(thawed(labels[str(key)]), {"name": expected["name"], "color": list(expected["color"]),
                                                      "code": {"scheme": scheme, "value": value, "meaning": meaning}})


class TestSegRead(SegCase):
    def test_binary_two_segments_shuffled_frames(self):
        masks = _masks(self.truth)
        path = self.seg(masks=masks)
        stored = pydicom.dcmread(path).PerFrameFunctionalGroupsSequence
        order = [int(item.SegmentIdentificationSequence[0].ReferencedSegmentNumber) for item in stored]
        self.assertNotEqual(order, sorted(order))                       # frames really shuffled
        image, report = medvol.read(path)
        values = image.meta.values
        self.assertEqual((values.kind, values.label_layout), ("label", "index"))
        self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
        self.assertEqual(image.array.shape, (5, 6, 5))                  # slices 1..5 of the CT (0 is empty)
        self.check_labels(values.labels, {1: 1, 2: 2})
        for number, mask in masks.items():
            _segment_error(self, image, self.truth, mask, image.array == number)
        self.assertTrue(any("empty" in w for w in report.warnings))     # slice 3 was omitted
        header = image.meta.source[0].header
        self.assertEqual(len(header["frames"]), len(stored))
        self.assertEqual(header["shared"]["SegmentationType"], "BINARY")

    def test_fractional(self):
        rng = np.random.default_rng(1)
        probability = rng.random(self.truth.shape) * _masks(self.truth)[1]
        image, _ = medvol.read(self.seg("fractional.dcm", {1: probability}, segmentation_type="FRACTIONAL"))
        values = image.meta.values
        self.assertEqual((values.kind, values.label_layout), ("probability", "channel"))
        self.assertAlmostEqual(values.scale, 1 / 255)
        self.assertEqual(image.array.dtype, np.uint8)                   # stored values, not rescaled
        self.assertEqual([(a.name, a.type) for a in image.meta.axes][0], ("c", "channel"))
        self.check_labels(values.labels, {0: 1})
        _segment_error(self, image, self.truth, probability > 0, image.array[0] > 0)
        expected = np.round(probability * 255)
        world = _voxel_world(image, image.array[0] > 0)
        ijk = np.round((np.linalg.inv(self.truth.affine_lps) @ np.vstack([world.T, np.ones(len(world))]))[:3]).astype(int)
        np.testing.assert_array_equal(image.array[0][image.array[0] > 0], expected[tuple(ijk)])

    def test_overlapping_segments_use_channels(self):
        masks = _masks(self.truth, overlap=True)
        image, report = medvol.read(self.seg("overlap.dcm", masks))
        values = image.meta.values
        self.assertEqual((values.kind, values.label_layout), ("label", "channel"))
        self.assertEqual(image.array.shape, (2, 5, 6, 5))
        self.check_labels(values.labels, {0: 1, 1: 2})
        for c, number in enumerate((1, 2)):
            _segment_error(self, image, self.truth, masks[number], image.array[c] > 0)
        self.assertTrue(any("overlap" in w for w in report.warnings))

    def test_label_map_seg(self):
        masks = _masks(self.truth)
        label_map = np.zeros(self.truth.shape, dtype=np.uint8)
        for number, mask in masks.items():
            label_map[mask] = number
        image, _ = medvol.read(self.seg("labelmap.dcm", {0: label_map}, segmentation_type="LABELMAP"))
        self.assertEqual(image.meta.values.label_layout, "index")
        for number, mask in masks.items():
            _segment_error(self, image, self.truth, mask, image.array == number)

    def test_seg_of_a_tilted_series(self):
        truth = phantom.sheared_truth()
        truth.dtype = "int16"
        series = self.dir / "tilted"
        phantom_dicom.write_ct_series(series, truth)
        masks = _masks(truth)
        image, report = medvol.read(phantom_dicom.write_seg(self.dir / "tilted.dcm", series, masks, truth=truth))
        self.assertEqual(report.choices["dicom_tilt"], "exact")
        for number, mask in masks.items():
            _segment_error(self, image, truth, mask, image.array == number)
        with self.assertRaises(GeometryError):
            medvol.read(self.dir / "tilted.dcm", dicom_tilt="error")


class TestSegWrite(SegCase):
    def via_mla(self, path):
        image, _ = medvol.read(path)
        medvol.write(image, self.dir / "x.mla")
        again, _ = medvol.read(self.dir / "x.mla")
        return image, again

    def test_seg_mla_seg_round_trip(self):
        rng = np.random.default_rng(2)
        cases = {
            "binary": (_masks(self.truth), "BINARY"),
            "overlap": (_masks(self.truth, overlap=True), "BINARY"),
            "fractional": ({1: rng.random(self.truth.shape) * _masks(self.truth)[1]}, "FRACTIONAL"),
        }
        label_map = np.zeros(self.truth.shape, dtype=np.uint8)
        for number, mask in _masks(self.truth).items():
            label_map[mask] = number
        cases["labelmap"] = ({0: label_map}, "LABELMAP")
        for name, (masks, kind) in cases.items():
            with self.subTest(case=name):
                path = self.seg(f"{name}.dcm", masks, segmentation_type=kind)
                image, again = self.via_mla(path)
                out = self.dir / f"out_{name}"
                report = medvol.write(again, out, format="dicom", reference=self.series)
                self.assertEqual(report.dropped, [])
                written_path = out / Path(path).name
                written, _ = medvol.read(written_path)
                np.testing.assert_array_equal(written.array, image.array)
                self.assertEqual(written.meta.values.to_dict(), image.meta.values.to_dict())
                np.testing.assert_allclose(lps_matrix(written.space()), lps_matrix(image.space()), atol=1e-9)
                original, result = pydicom.dcmread(path), pydicom.dcmread(written_path)
                for a, b in zip(original.SegmentSequence, result.SegmentSequence):
                    self.assertEqual(a, b)                                  # codes, names, colours, category, tracking
                for keyword in ("SOPInstanceUID", "SeriesInstanceUID", "SeriesDescription", "ContentLabel",
                                "ContentDescription", "ContentDate", "Manufacturer", "SegmentationType"):
                    self.assertEqual(original.get(keyword), result.get(keyword), keyword)

    def test_label_map_from_another_source_on_a_shuffled_reference(self):
        ct, _ = medvol.read(self.series)
        masks = _masks(self.truth)
        label = np.zeros(ct.array.shape, dtype=np.uint8)
        ijk = np.unravel_index(ct.array.astype(np.int64), self.truth.shape, order="F")   # each CT voxel's file index
        for number, mask in masks.items():
            label[mask[ijk]] = number * 2                                # values 2 and 4: renumbered to 1, 2
        meta = Meta(axes=[a.to_dict() for a in ct.meta.axes], spaces=[ct.space().to_dict()],
                    values={"kind": "label", "labels": {"2": {"name": "liver", "code": {"scheme": "SCT", "value": "10200004", "meaning": "Liver"}},
                                                        "4": {"name": "spleen", "color": [10, 200, 30]}}},
                    shape=label.shape)
        image = medvol.Image(label, meta)
        files = [str(p) for p in self.series.iterdir()]
        random.Random(0).shuffle(files)
        report = dicom_seg.write_seg(image, str(self.dir / "new.dcm"), medvol.Policy(), reference=files)
        self.assertEqual(report.choices["segment_numbers"], "renumbered")
        self.assertTrue(any("no code" in w for w in report.warnings))
        written, _ = medvol.read(self.dir / "new.dcm")
        self.assertEqual(written.array.shape, (7, 6, 5))                 # all frames kept (omit_empty_frames=False)
        labels = written.meta.values.labels
        self.assertEqual(labels["1"]["name"], "liver")
        self.assertEqual(thawed(labels["1"]["code"]), {"scheme": "SCT", "value": "10200004", "meaning": "Liver"})
        self.assertEqual(list(labels["2"]["color"]), [10, 200, 30])
        for number, mask in masks.items():
            _segment_error(self, written, self.truth, mask, written.array == number)

    def test_geometry_must_match_the_reference(self):
        image, _ = medvol.read(self.seg())
        moved = image.space().matrix()
        moved[:3, -1] += 0.5
        shifted = medvol.Image(image.array, image.meta.copy())
        shifted.meta.spaces = [dict(image.space().to_dict(), from_array={"array_axes": ["z", "y", "x"], "matrix": moved.tolist()})]
        with self.assertRaisesRegex(GeometryError, "reference grid"):
            medvol.write(shifted, self.dir / "out", format="dicom", reference=self.series)
        flipped_matrix = image.space().matrix()
        flipped_matrix[:3, -1] += flipped_matrix[:3, 2] * (image.array.shape[2] - 1)
        flipped_matrix[:3, 2] *= -1
        flipped = medvol.Image(image.array[:, :, ::-1].copy(), image.meta.copy())
        flipped.meta.spaces = [dict(image.space().to_dict(), from_array={"array_axes": ["z", "y", "x"], "matrix": flipped_matrix.tolist()})]
        with self.assertRaisesRegex(GeometryError, "reference grid"):
            medvol.write(flipped, self.dir / "out", format="dicom", reference=self.series)
        missing = medvol.Image(image.array, image.meta.copy())
        with self.assertRaisesRegex(ValueError, "reference="):
            medvol.write(missing, self.dir / "out", format="dicom")


if __name__ == "__main__":
    unittest.main()
