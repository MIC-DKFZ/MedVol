import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import medvol
import medvol.registry
import phantom
import phantom_metaimage as pmi
from medvol.plugins.metaimage import MetaImagePlugin
from medvol.report import LossError

medvol.registry.register(MetaImagePlugin())


def _number_or_text(text):
    try:
        return [float(v) for v in text.split()]
    except ValueError:
        return text


def _header_values(path):
    """Header as (key, value) pairs: numbers compared by value (MLArray spec 9), text verbatim."""
    lines, _ = pmi.read_metaimage(path)
    return [(key, _number_or_text(value)) for key, value in lines]


class MetaImageCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def round_trip(self, path, **options):
        """X -> .mla -> X; returns the path of the rewritten file."""
        image, _ = medvol.read(path, **options)
        mla = self.dir / (Path(path).stem + ".mla")
        medvol.write(image, mla)
        again, _ = medvol.read(mla)
        out = self.dir / "out" / Path(path).name
        out.parent.mkdir(exist_ok=True)
        medvol.write(again, out, **options)
        return out

    def assert_same_file(self, out, path, ignore=()):
        expected = [(k, v) for k, v in _header_values(path) if k not in ignore]
        actual = [(k, v) for k, v in _header_values(out) if k not in ignore]
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(pmi.read_metaimage(out)[1], pmi.read_metaimage(path)[1])


class TestRead(MetaImageCase):
    def test_world_positions(self):
        truth = phantom.Truth()
        for name, kwargs in {"plain.mha": {}, "detached.mhd": {}, "compressed.mha": {"compressed": True},
                             "msb.mha": {"msb": True}, "rotation.mha": {"direction_key": "Rotation"},
                             "orientation.mhd": {"direction_key": "Orientation", "compressed": True}}.items():
            with self.subTest(file=name):
                path = pmi.write_metaimage(self.dir / name, truth, **kwargs)
                self.assertLess(pmi.file_world_error(path, truth), 1e-9)      # the phantom itself
                image, _ = medvol.read(path)
                self.assertLess(phantom.world_error(image, truth), 1e-4)
                self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
                self.assertEqual(image.meta.source[0].index_map["axis_order"], (2, 1, 0))
                self.assertEqual(image.meta.source[0].format, "metaimage")

    def test_transform_matrix_is_one_axis_per_triple(self):
        """KB ch04 3.2 / pitfall P13: a row-major D is read as D^T, as ITK does."""
        truth = phantom.Truth()
        path = pmi.write_metaimage(self.dir / "row.mha", truth, row_major=True)
        image, _ = medvol.read(path)
        self.assertGreater(phantom.world_error(image, truth), 1.0)
        self.assertAlmostEqual(phantom.world_error(image, truth), pmi.file_world_error(path, truth), places=6)

    def test_origin_wins_over_offset_and_position(self):
        """KB ch04 3.3 / e02 B: Position -> Offset -> Origin, the last present wins whatever the line order."""
        truth = phantom.Truth()
        decoys = [("Origin", None), ("Offset", (1.0, 1.0, 1.0)), ("Position", (2.0, 2.0, 2.0))]
        path = pmi.write_metaimage(self.dir / "origins.mha", truth, origins=decoys)
        image, report = medvol.read(path)
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assertTrue(any("Origin is used" in w for w in report.warnings))
        self.assertEqual(report.choices["metaimage_geometry"]["origin"], "Origin")
        # without Origin, Offset beats Position
        path = pmi.write_metaimage(self.dir / "offset.mha", truth, origins=[("Offset", None), ("Position", (2.0, 2.0, 2.0))])
        self.assertLess(phantom.world_error(medvol.read(path)[0], truth), 1e-4)

    def test_anatomical_orientation_is_never_used(self):
        truth = phantom.Truth()
        path = pmi.write_metaimage(self.dir / "ao.mha", truth, extra=[("AnatomicalOrientation", "LAS"),
                                                                       ("Comment", "decoy 0.70"), ("MyKey", "1.50")])
        image, _ = medvol.read(path)
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        header = image.meta.source[0].header
        self.assertEqual(header["AnatomicalOrientation"], "LAS")
        self.assertEqual(header["MyKey"], "1.50")                       # unknown keys stay text
        self.assertEqual(header["TransformMatrix"][0], float(pmi.geometry_lines(truth)["TransformMatrix"].split()[0]))
        self.assertEqual(list(header)[:3], ["ObjectType", "NDims", "BinaryData"])

    def test_ras_world_frame_is_a_row_sign_change(self):
        truth = phantom.Truth()
        path = pmi.write_metaimage(self.dir / "a.mha", truth)
        lps, _ = medvol.read(path)
        ras, _ = medvol.read(path, world_frame="ras")
        np.testing.assert_allclose(ras.space().matrix(), np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps.space().matrix())
        self.assertLess(phantom.world_error(ras, truth), 1e-4)

    def test_multi_channel(self):
        truth = phantom.Truth(shape=(5, 6, 7, 3))
        path = pmi.write_metaimage(self.dir / "rgb.mha", truth, channels=True)
        image, _ = medvol.read(path)
        self.assertEqual([(a.name, a.type) for a in image.meta.axes],
                         [("z", "space"), ("y", "space"), ("x", "space"), ("c", "channel")])
        self.assertEqual(image.array.shape, (7, 6, 5, 3))
        self.assertEqual(image.meta.source[0].index_map["axis_order"], (2, 1, 0, 3))
        self.assertEqual(list(image.space().from_array.array_axes), ["x", "y", "z"])
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assert_same_file(self.round_trip(path), path)

    def test_list_of_slice_files(self):
        truth = phantom.Truth(dtype="int16")
        path = pmi.write_metaimage(self.dir / "full.mhd", truth)
        lines, volume = pmi.read_metaimage(path)
        names = []
        for k in range(truth.shape[2]):
            names.append(f"slice{k:02d}.raw")
            (self.dir / names[-1]).write_bytes(np.asarray(volume[:, :, k], dtype="<i2").tobytes(order="F"))
        text = "".join(f"{key} = {value}\n" for key, value in lines if key != "ElementDataFile")
        (self.dir / "list.mhd").write_text(text + "ElementDataFile = LIST\n" + "".join(n + "\n" for n in names))
        image, _ = medvol.read(self.dir / "list.mhd")
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assertEqual(list(image.meta.source[0].files), ["list.mhd"] + names)
        out = self.round_trip(self.dir / "list.mhd")
        again, _ = medvol.read(out)
        np.testing.assert_array_equal(again.array, image.array)
        self.assertEqual(again.meta.source[0].header["ElementDataFile"], "LIST")

    def test_time_axis(self):
        truth = phantom.Truth(shape=(5, 6, 7, 3), dtype="int16")
        path = pmi.write_metaimage(self.dir / "t.mha", truth)
        image, _ = medvol.read(path)
        self.assertEqual([(a.name, a.type) for a in image.meta.axes][0], ("t", "time"))
        self.assertEqual((image.meta.axes[0].start, image.meta.axes[0].step), (0.0, 1.0))
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assertEqual(image.space().matrix().shape, (4, 4))          # time never enters the space


class TestRoundTrip(MetaImageCase):
    def test_same_format_round_trip_is_exact(self):
        truth = phantom.Truth()
        cases = {
            "plain.mha": {}, "compressed.mha": {"compressed": True}, "msb.mha": {"msb": True},
            "detached.mhd": {}, "zdetached.mhd": {"compressed": True},
            "decoys.mha": {"origins": [("Origin", None), ("Offset", (1.0, 1.0, 1.0))], "direction_key": "Rotation",
                           "extra": [("AnatomicalOrientation", "LAS"), ("Comment", "phantom 0.7"), ("Custom", "a = b")]},
            "float.mha": {"truth": phantom.Truth(dtype="float32")},
            "time.mha": {"truth": phantom.Truth(shape=(5, 6, 7, 3))},
        }
        for name, kwargs in cases.items():
            with self.subTest(file=name):
                kwargs = dict(kwargs)
                path = pmi.write_metaimage(self.dir / name, kwargs.pop("truth", truth), **kwargs)
                out = self.round_trip(path)
                self.assert_same_file(out, path)
                self.assertLess(pmi.file_world_error(out, truth if "truth" not in cases[name] else cases[name]["truth"]), 1e-9)

    def test_minimal_ascii_and_header_size_files(self):
        """Absent fields stay absent; ASCII data and a skipped raw-file header are read."""
        volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)          # C order [k, j, i]
        minimal = self.dir / "minimal.mha"
        minimal.write_bytes(b"NDims = 3\nDimSize = 4 3 2\nElementType = MET_SHORT\nElementDataFile = LOCAL\n"
                            + volume.astype("<i2").tobytes())
        ascii_file = self.dir / "ascii.mha"
        ascii_file.write_text("ObjectType = Image\nNDims = 3\nBinaryData = False\nDimSize = 4 3 2\n"
                              "ElementType = MET_SHORT\nElementDataFile = LOCAL\n"
                              + "\n".join(" ".join(str(v) for v in row) for row in volume.reshape(-1, 4)) + "\n")
        skipped = self.dir / "skipped.mhd"
        skipped.write_text("ObjectType = Image\nNDims = 3\nBinaryData = True\nHeaderSize = -1\nDimSize = 4 3 2\n"
                           "ElementType = MET_SHORT\nElementDataFile = skipped.raw\n")
        (self.dir / "skipped.raw").write_bytes(b"\x07" * 13 + volume.astype("<i2").tobytes())
        for path in (minimal, ascii_file, skipped):
            with self.subTest(file=path.name):
                image, _ = medvol.read(path)
                np.testing.assert_array_equal(image.array, volume)
                np.testing.assert_array_equal(image.space().matrix(), np.eye(4))
        out = self.round_trip(minimal)
        self.assertEqual([key for key, _ in pmi.read_metaimage(out)[0]], ["NDims", "DimSize", "ElementType", "ElementDataFile"])
        out = self.round_trip(ascii_file)
        self.assertEqual(medvol.read(out)[0].meta.source[0].header["BinaryData"], False)
        np.testing.assert_array_equal(medvol.read(out)[0].array, volume)

    def test_detached_layout_and_compression_are_kept(self):
        path = pmi.write_metaimage(self.dir / "a.mhd", phantom.Truth(), compressed=True)
        out = self.round_trip(path)
        self.assertEqual(dict(pmi.read_metaimage(out)[0])["ElementDataFile"], "a.zraw")
        self.assertTrue((out.parent / "a.zraw").exists())
        default = self.dir / "out" / "default.mha"
        medvol.write(medvol.read(path)[0], default, layout="default")
        header = dict(pmi.read_metaimage(default)[0])
        self.assertEqual((header["CompressedData"], header["ElementDataFile"]), ("False", "LOCAL"))

    def test_core_owned_fields_regenerate_to_the_original_values(self):
        """Spec 4: regenerating the core-owned fields from an unchanged core reproduces them."""
        cases = {"plain.mha": (phantom.Truth(), {}),
                 "synonyms.mha": (phantom.Truth(), {"origins": [("Origin", None), ("Position", None)], "direction_key": "Orientation"}),
                 "rgb.mhd": (phantom.Truth(shape=(5, 6, 7, 2)), {"channels": True}),
                 "time.mha": (phantom.Truth(shape=(5, 6, 7, 3)), {})}
        for name, (truth, kwargs) in cases.items():
            with self.subTest(case=name):
                path = pmi.write_metaimage(self.dir / name, truth, **kwargs)
                image, _ = medvol.read(path)
                out = self.dir / "out" / name
                out.parent.mkdir(exist_ok=True)
                with mock.patch.object(MetaImagePlugin, "_geometry_unchanged", return_value=False):
                    medvol.write(image, out)
                original, regenerated = _header_values(path), _header_values(out)
                self.assertEqual([k for k, _ in regenerated], [k for k, _ in original])
                for (key, value), (_, again) in zip(original, regenerated):
                    if isinstance(value, list):
                        np.testing.assert_allclose(again, value, rtol=1e-12, atol=1e-15, err_msg=key)
                    elif key != "ElementDataFile":
                        self.assertEqual(again, value, key)
                self.assertLess(pmi.file_world_error(out, truth), 1e-9)

    def test_sheared_grid_is_reported(self):
        truth = phantom.sheared_truth()
        nifti, _ = medvol.read(phantom.write_nifti(self.dir / "tilt.nii.gz", truth))
        report = medvol.write(nifti, self.dir / "tilt.mha")
        self.assertTrue(any(d.startswith("geometry: sheared") for d in report.dropped))
        self.assertTrue(any("closest orthonormal" in w for w in report.warnings))
        lines, _ = pmi.read_metaimage(self.dir / "tilt.mha")
        direction = np.array(dict(lines)["TransformMatrix"].split(), dtype=float).reshape(3, 3).T
        np.testing.assert_allclose(direction.T @ direction, np.eye(3), atol=1e-12)
        with self.assertRaises(LossError):
            medvol.write(nifti, self.dir / "tilt2.mha", on_loss="error")

    def test_changed_data_drops_element_min_max(self):
        path = pmi.write_metaimage(self.dir / "a.mha", extra=[("ElementMin", "0"), ("ElementMax", "209")])
        image, _ = medvol.read(path)
        out = self.dir / "out.mha"
        medvol.write(image, out)
        self.assertEqual(dict(pmi.read_metaimage(out)[0])["ElementMax"], "209")
        image.array[0, 0, 0] += 1
        report = medvol.write(image, out)
        self.assertNotIn("ElementMax", dict(pmi.read_metaimage(out)[0]))
        self.assertTrue(any("ElementMax dropped" in w for w in report.warnings))


class TestCrossFormat(MetaImageCase):
    def test_nifti_to_metaimage_keeps_world_positions(self):
        truth = phantom.Truth()
        nifti = phantom.write_nifti(self.dir / "a.nii.gz", truth, extra={"descrip": "from nifti", "cal_max": 9.0})
        for name in ("a.mha", "a.mhd"):
            with self.subTest(file=name):
                report = medvol.convert(nifti, self.dir / name)
                self.assertLess(pmi.file_world_error(self.dir / name, truth), 1e-4)
                again, _ = medvol.read(self.dir / name)
                self.assertLess(phantom.world_error(again, truth), 1e-4)
                keys = [key for key, _ in pmi.read_metaimage(self.dir / name)[0]]
                self.assertEqual(keys, ["ObjectType", "NDims", "BinaryData", "BinaryDataByteOrderMSB", "CompressedData",
                                        "TransformMatrix", "Offset", "ElementSpacing", "DimSize", "ElementType",
                                        "ElementDataFile"])
                self.assertEqual(report.dropped, [])

    def test_metaimage_to_nifti_keeps_world_positions(self):
        truth = phantom.Truth()
        path = pmi.write_metaimage(self.dir / "a.mha", truth, extra=[("Comment", "x")])
        medvol.convert(path, self.dir / "a.nii.gz")
        image, _ = medvol.read(self.dir / "a.nii.gz")
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        self.assertNotIn("Comment", image.meta.source[0].header)


if __name__ == "__main__":
    unittest.main()
