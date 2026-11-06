import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nrrd
import numpy as np
from mlarray import MLArray

import medvol
import medvol.registry
import phantom
import phantom_nrrd
from medvol.geometry import GeometryError
from medvol.plugins.nrrd import NrrdPlugin

medvol.registry.register(NrrdPlugin())

STANDARD_FIELDS = {"type", "dimension", "space dimension", "space", "sizes", "space directions", "kinds", "endian",
                   "encoding", "min", "max", "old min", "old max", "content", "sample units", "spacings",
                   "thicknesses", "axis mins", "axis maxs", "centerings", "labels", "units", "space units",
                   "space origin", "measurement frame", "data file"}


def _header_text(path):
    """The header exactly as stored (magic, comments, fields, key-value pairs)."""
    return Path(path).read_bytes().split(b"\n\n", 1)[0].decode("ascii")


def _data(path):
    return nrrd.read(str(path), index_order="F")[0]


def _vector_first(**kwargs):
    """3-component vector image with the component axis first (fastest): review finding H12."""
    return phantom_nrrd.AxisTruth(shape=(3, 5, 6, 7), spatial=(1, 2, 3)), dict(
        kinds=["vector", "domain", "domain", "domain"], **kwargs)


def _time_cell(centerings=("???", "???", "???", "cell")):
    extra = {"spacings": [np.nan, np.nan, np.nan, 2.0], "axis mins": [np.nan, np.nan, np.nan, 10.0]}
    if centerings is not None:
        extra["centerings"] = list(centerings)
    return phantom_nrrd.AxisTruth(shape=(5, 6, 7, 3)), dict(kinds=["domain"] * 3 + ["time"], extra=extra)


class NrrdCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def round_trip(self, path, suffix=None, **options):
        """X -> .mla -> X; returns the path of the rewritten file."""
        image, _ = medvol.read(path, **options)
        stem = Path(path).name.split(".")[0]
        mla = self.dir / f"{stem}.mla"
        medvol.write(image, mla)
        again, _ = medvol.read(mla)
        out = self.dir / f"{stem}_rt{suffix or Path(path).suffix}"
        medvol.write(again, out, **options)
        return out


class TestRead(NrrdCase):
    def test_world_positions_of_all_spaces(self):
        truth = phantom.Truth()
        for space in phantom.NRRD_FRAMES:
            for frame in ("dicom_patient", "ras"):
                with self.subTest(space=space, world_frame=frame):
                    path = phantom.write_nrrd(self.dir / f"{space}.nrrd", truth, space)
                    image, report = medvol.read(path, world_frame=frame)
                    self.assertLess(phantom.world_error(image, truth), 1e-4)
                    self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x"])
                    self.assertEqual(image.meta.source[0].index_map["axis_order"], (2, 1, 0))
                    self.assertEqual(report.choices["nrrd_space"], f"{space} -> {frame}")
        for space in ("3D-left-handed", "LPS", "RAS", "LAS"):
            with self.subTest(space=space):
                truth = phantom_nrrd.AxisTruth()
                image, _ = medvol.read(phantom_nrrd.write(self.dir / "s.nrrd", truth, space=space))
                self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_spaces_without_anatomical_meaning_are_reported(self):
        for space in ("scanner-xyz", "3D-right-handed"):
            with self.subTest(space=space):
                image, report = medvol.read(phantom.write_nrrd(self.dir / "a.nrrd", space=space))
                self.assertTrue(any("taken as LPS" in w for w in report.warnings))
        _, report = medvol.read(phantom.write_nrrd(self.dir / "b.nrrd", space="left-anterior-superior"))
        self.assertEqual(report.warnings, [])

    def test_unknown_space_is_rejected(self):
        path = phantom.write_nrrd(self.dir / "a.nrrd", space="left-posterior-superior")
        text = Path(path).read_bytes().replace(b"space: left-posterior-superior", b"space: posterior-left-up")
        Path(path).write_bytes(text)
        with self.assertRaisesRegex(GeometryError, "Unknown NRRD space"):
            medvol.read(path)

    def test_time_spaces(self):
        truth = phantom_nrrd.AxisTruth(shape=(5, 6, 7, 3))
        for space in phantom_nrrd.TIME_SPACES:
            with self.subTest(space=space):
                path = phantom_nrrd.write(self.dir / "t.nrrd", truth, ["domain"] * 3 + ["time"], space=space,
                                          time_axis=3, time=(4.5, 0.25))
                image, _ = medvol.read(path)
                self.assertEqual([(a.name, a.type) for a in image.meta.axes],
                                 [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")])
                self.assertEqual((image.meta.axes[0].start, image.meta.axes[0].step), (4.5, 0.25))
                self.assertEqual(image.space().matrix().shape, (4, 4))
                self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_coupled_space_and_time_are_rejected(self):
        truth = phantom_nrrd.AxisTruth(shape=(5, 6, 7, 3))
        rows = np.zeros((4, 4))
        rows[:3, :3] = np.diag([1.0, 1.0, 1.0])
        rows[3] = [0.5, 0.0, 0.0, 2.0]                # time axis that also moves in x
        path = phantom_nrrd.write(self.dir / "t.nrrd", truth, ["domain"] * 3 + ["time"], space="RAST",
                                  time_axis=3, directions=rows)
        with self.assertRaisesRegex(GeometryError, "couples space and time"):
            medvol.read(path)

    def test_vector_first_file_has_correct_geometry(self):
        """Review finding H12: the vector axis never enters the space matrix."""
        truth, kwargs = _vector_first()
        image, _ = medvol.read(phantom_nrrd.write(self.dir / "v.nrrd", truth, **kwargs))
        self.assertEqual([(a.name, a.type) for a in image.meta.axes],
                         [("z", "space"), ("y", "space"), ("x", "space"), ("c", "component")])
        self.assertEqual(image.array.shape, (7, 6, 5, 3))
        self.assertEqual(list(image.space().from_array.array_axes), ["x", "y", "z"])
        self.assertEqual(image.space().matrix().shape, (4, 4))
        self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_kinds_map_to_axis_types(self):
        cases = {"RGB-color": "channel", "covariant-vector": "component", "3D-symmetric-matrix": "component",
                 "list": "other", "time": "time", "domain": "other"}
        for kind, axis_type in cases.items():
            with self.subTest(kind=kind):
                size = 6 if kind == "3D-symmetric-matrix" else 3
                truth = phantom_nrrd.AxisTruth(shape=(5, 6, 7, size))
                image, _ = medvol.read(phantom_nrrd.write(self.dir / "k.nrrd", truth, ["domain"] * 3 + [kind]))
                self.assertEqual(image.meta.axes[0].type, axis_type)
                self.assertEqual(list(image.space().from_array.array_axes), ["x", "y", "z"])
                self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_cell_centering_is_converted_to_voxel_centres(self):
        for centerings, start, choice in ((("???", "???", "???", "cell"), 11.0, False),
                                          (("???", "???", "???", "node"), 10.0, False),
                                          (None, 11.0, True)):
            with self.subTest(centerings=centerings):
                truth, kwargs = _time_cell(centerings)
                image, report = medvol.read(phantom_nrrd.write(self.dir / "c.nrrd", truth, **kwargs))
                time = image.meta.axes[0]
                self.assertEqual((time.name, time.type, time.start, time.step), ("t", "time", start, 2.0))
                self.assertEqual("centering axis 3" in report.choices, choice)
                self.assertLess(phantom.world_error(image, truth), 1e-4)

    def test_axis_mins_and_maxs_without_spacing(self):
        truth = phantom_nrrd.AxisTruth(shape=(5, 6, 7, 4))
        extra = {"axis mins": [np.nan] * 3 + [0.0], "axis maxs": [np.nan] * 3 + [1.0],
                 "centerings": ["???"] * 3 + ["cell"]}
        image, _ = medvol.read(phantom_nrrd.write(self.dir / "c.nrrd", truth, ["domain"] * 3 + ["list"], extra=extra))
        self.assertEqual((image.meta.axes[0].start, image.meta.axes[0].step), (0.125, 0.25))   # NRRD spec example

    def test_no_space_means_index_space(self):
        truth = phantom_nrrd.AxisTruth()
        header = {"spacings": [0.5, 1.0, 2.0], "axis mins": [10.0, -20.0, 30.0], "encoding": "raw"}
        image, report = medvol.read(phantom_nrrd.write_plain(self.dir / "a.nrrd", truth, header))
        self.assertEqual(image.meta.spaces, ())
        self.assertEqual([(a.name, a.type) for a in image.meta.axes], [("z", "space"), ("y", "space"), ("x", "space")])
        self.assertEqual(image.meta.source[0].header["fields"]["spacings"], (0.5, 1.0, 2.0))
        self.assertTrue(any("index space only" in w for w in report.warnings))

    def test_unnamed_space_is_not_a_patient_space(self):
        truth = phantom_nrrd.AxisTruth()
        header = {"space dimension": 3, "space directions": np.diag([0.5, 1.0, 2.0]), "space origin": [1.0, 2.0, 3.0],
                  "space units": ["um", "um", "um"], "encoding": "raw"}
        image, report = medvol.read(phantom_nrrd.write_plain(self.dir / "a.nrrd", truth, header))
        space = image.space()
        self.assertEqual([(a.unit, a.increases_toward) for a in space.axes], [("micrometer", None)] * 3)
        np.testing.assert_array_equal(space.matrix()[:3, :3], np.diag([0.5, 1.0, 2.0]))
        self.assertIn("not a patient space", report.choices["nrrd_space"])

    def test_header_is_readable_and_complete(self):
        mframe = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        extra = {"measurement frame": mframe, "content": "phantom 0.7", "Segment0_Name": "liver",
                 "DWMRI_b-value": "1000", "min": 0.0, "max": 209.0}
        path = phantom.write_nrrd(self.dir / "a.nrrd", space="right-anterior-superior", extra=extra)
        image, report = medvol.read(path)
        header = image.meta.source[0].header
        self.assertEqual(header["magic"], "NRRD0005")
        self.assertEqual(image.meta.source[0].format_version, "5")
        self.assertEqual(header["comments"][0], " This NRRD file was generated by pynrrd")
        self.assertEqual(dict(header["key_values"]), {"Segment0_Name": "liver", "DWMRI_b-value": "1000"})
        self.assertEqual(header["fields"]["content"], "phantom 0.7")
        self.assertEqual(header["fields"]["sizes"], (5, 6, 7))
        self.assertIsInstance(header["fields"]["dimension"], int)
        # measurement frame as written: entry i is column i of the frame matrix
        self.assertEqual(header["fields"]["measurement frame"], ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)))
        self.assertIn("measurement_frame", report.choices)
        np.testing.assert_array_equal(image.array, phantom.Truth().volume().T)
        self.assertEqual(list(header["fields"])[:4], ["type", "dimension", "space", "sizes"])


class TestRoundTrip(NrrdCase):
    def cases(self):
        vector, vector_kwargs = _vector_first()
        cell, cell_kwargs = _time_cell()
        timed = phantom_nrrd.AxisTruth(shape=(5, 6, 7, 3))
        cases = {f"{space}.nrrd": (phantom_nrrd.AxisTruth(), {"space": space}) for space in phantom_nrrd.SPACES}
        cases.update({
            "gzip.nrrd": (phantom_nrrd.AxisTruth(), {"encoding": "gzip"}),
            "bzip2.nrrd": (phantom_nrrd.AxisTruth(), {"encoding": "bzip2"}),
            "ascii.nrrd": (phantom_nrrd.AxisTruth(), {"encoding": "ascii"}),
            "big.nrrd": (phantom_nrrd.AxisTruth(dtype="int16"), {"big_endian": True}),
            "vector.nrrd": (vector, vector_kwargs),
            "cell.nrrd": (cell, cell_kwargs),
            "rast.nrrd": (timed, {"kinds": ["domain"] * 3 + ["time"], "space": "RAST", "time_axis": 3,
                                  "time": (1.5, 0.5), "extra": {"space units": ["mm", "mm", "mm", "s"]}}),
            "keys.nrrd": (phantom_nrrd.AxisTruth(), {"space": "RAS", "extra": {
                "measurement frame": np.arange(9.0).reshape(3, 3), "content": "x", "min": 0.0, "max": 209.0,
                "my key": "a value:=with=signs ", "labels": ["i", "j", "k"], "thicknesses": [np.nan, np.nan, 2.5]}}),
        })
        return cases

    def test_same_format_round_trip_is_exact(self):
        for name, (truth, kwargs) in self.cases().items():
            with self.subTest(file=name):
                path = phantom_nrrd.write(self.dir / name, truth, **kwargs)
                out = self.round_trip(path)
                self.assertEqual(_header_text(out), _header_text(path))
                np.testing.assert_array_equal(_data(out), _data(path))
                self.assertEqual(_data(out).dtype, _data(path).dtype)

    def test_core_owned_fields_regenerate_to_the_original_values(self):
        """Spec 4: regenerating the core-owned fields from an unchanged core reproduces them."""
        cases = self.cases()
        cases["plain.nrrd"] = (phantom_nrrd.AxisTruth(), "plain")
        for name, (truth, kwargs) in cases.items():
            for frame in ("dicom_patient", "ras"):
                with self.subTest(file=name, world_frame=frame):
                    if kwargs == "plain":
                        path = phantom_nrrd.write_plain(self.dir / name, truth, {
                            "spacings": [0.5, 1.0, 2.0], "axis mins": [10.0, -20.0, 30.0], "encoding": "raw",
                            "centerings": ["cell", "node", "???"], "labels": ["a", "b", "c"]})
                    else:
                        path = phantom_nrrd.write(self.dir / name, truth, **kwargs)
                    image, _ = medvol.read(path, world_frame=frame)
                    out = self.dir / f"regen_{name}"
                    with mock.patch.object(NrrdPlugin, "_geometry_unchanged", return_value=False):
                        report = medvol.write(image, out)
                    self.assertNotIn("geometry", report.choices)
                    self.assertEqual(_header_text(out), _header_text(path))
                    np.testing.assert_array_equal(_data(out), _data(path))

    def test_detached_header(self):
        truth = phantom.Truth()
        phantom.write_nrrd(self.dir / "d.nrrd", truth, detached=True)       # pynrrd: header d.nhdr, data d.nrrd
        path = self.dir / "d.nhdr"
        image, _ = medvol.read(path)
        self.assertEqual(image.meta.source[0].files, ("d.nhdr", "d.nrrd"))
        self.assertLess(phantom.world_error(image, truth), 1e-4)
        out = self.round_trip(path)
        self.assertTrue((self.dir / "d_rt.raw").exists())
        self.assertIn("data file: d_rt.raw", _header_text(out))
        self.assertEqual(_header_text(out).replace("d_rt.raw", "d.nrrd"), _header_text(path))
        np.testing.assert_array_equal(_data(out), _data(path))
        attached = self.dir / "attached.nrrd"
        medvol.write(image, attached)
        self.assertNotIn("data file", _header_text(attached))
        again, _ = medvol.read(attached)
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_per_axis_fields_follow_their_axis_after_a_reorder(self):
        """Review finding M16: per-axis fields are never copied stale."""
        truth, kwargs = _vector_first(extra={"labels": ["vec", "li", "lj", "lk"], "thicknesses": [np.nan, 1.5, 2.5, 3.5],
                                             "units": ["mm/s", "", "", ""]})
        image, _ = medvol.read(phantom_nrrd.write(self.dir / "v.nrrd", truth, **kwargs))
        self.assertEqual([a.name for a in image.meta.axes], ["z", "y", "x", "c"])
        meta = image.meta.copy()
        order = (3, 0, 1, 2)                                           # c, z, y, x: vector axis last in NRRD
        meta._set_shape(image.array.transpose(order).shape, axes=[image.meta.axes[i] for i in order])
        moved = medvol.Image(image.array.transpose(order), meta)
        out = self.dir / "moved.nrrd"
        medvol.write(moved, out)
        header = nrrd.read_header(str(out))
        self.assertEqual(header["kinds"], ["domain", "domain", "domain", "vector"])
        self.assertEqual(header["labels"], ["li", "lj", "lk", "vec"])
        self.assertEqual(header["units"], ["", "", "", "mm/s"])
        np.testing.assert_array_equal(header["thicknesses"][:3], [1.5, 2.5, 3.5])
        self.assertTrue(np.isnan(header["thicknesses"][3]))
        self.assertTrue(np.all(np.isnan(header["space directions"][3])))
        again, _ = medvol.read(out)
        self.assertEqual([(a.name, a.type) for a in again.meta.axes],
                         [("c", "component"), ("z", "space"), ("y", "space"), ("x", "space")])
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_reordered_time_axis_keeps_its_sampling(self):
        truth, kwargs = _time_cell()
        image, _ = medvol.read(phantom_nrrd.write(self.dir / "c.nrrd", truth, **kwargs))
        meta = image.meta.copy()
        order = (1, 2, 3, 0)                                            # time becomes NRRD axis 0
        meta._set_shape(image.array.transpose(order).shape, axes=[image.meta.axes[i] for i in order])
        out = self.dir / "moved.nrrd"
        medvol.write(medvol.Image(image.array.transpose(order), meta), out)
        header = nrrd.read_header(str(out))
        self.assertEqual(header["kinds"], ["time", "domain", "domain", "domain"])
        self.assertEqual(header["centerings"], ["cell", "???", "???", "???"])
        self.assertEqual((header["spacings"][0], header["axis mins"][0]), (2.0, 10.0))
        again, _ = medvol.read(out)
        self.assertEqual((again.meta.axes[-1].start, again.meta.axes[-1].step), (11.0, 2.0))
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_changed_data_drops_min_max(self):
        path = phantom.write_nrrd(self.dir / "a.nrrd", extra={"min": 0.0, "max": 209.0})
        image, _ = medvol.read(path)
        image.array[0, 0, 0] += 1
        report = medvol.write(image, self.dir / "b.nrrd")
        header = nrrd.read_header(str(self.dir / "b.nrrd"))
        self.assertNotIn("min", header)
        self.assertTrue(any("max dropped" in w for w in report.warnings))

    def test_layout_policy(self):
        path = phantom_nrrd.write(self.dir / "a.nrrd", phantom_nrrd.AxisTruth(dtype="int16"), big_endian=True,
                                  encoding="bzip2")
        image, report = medvol.read(path)
        self.assertTrue(image.array.dtype.isnative)
        self.assertIn("byte_order", report.choices)
        medvol.write(image, self.dir / "default.nrrd", layout="default")
        header = nrrd.read_header(str(self.dir / "default.nrrd"))
        self.assertEqual((header["encoding"], header["endian"]), ("gzip", "little"))
        np.testing.assert_array_equal(_data(self.dir / "default.nrrd"), _data(path))
        medvol.write(image, self.dir / "source.nrrd")
        header = nrrd.read_header(str(self.dir / "source.nrrd"))
        self.assertEqual((header["encoding"], header["endian"]), ("bzip2", "big"))

    def test_unnamed_space_round_trips_and_new_ones_are_reported(self):
        truth = phantom_nrrd.AxisTruth()
        header = {"space dimension": 3, "space directions": np.diag([0.5, 1.0, 2.0]), "space origin": [1.0, 2.0, 3.0],
                  "space units": ["um", "um", "um"], "encoding": "raw"}
        path = phantom_nrrd.write_plain(self.dir / "a.nrrd", truth, header)
        image, _ = medvol.read(path)
        with mock.patch.object(NrrdPlugin, "_geometry_unchanged", return_value=False):
            report = medvol.write(image, self.dir / "b.nrrd")
        self.assertEqual(report.dropped, [])
        self.assertEqual(_header_text(self.dir / "b.nrrd"), _header_text(path))
        stage = MLArray(image.array, axes=["z", "y", "x"], spaces=[image.space().to_dict()])
        report = medvol.write(medvol.Image.from_mlarray(stage), self.dir / "c.nrrd")
        self.assertTrue(any("not a patient space" in d for d in report.dropped))
        with self.assertRaises(medvol.LossError):
            medvol.write(medvol.Image.from_mlarray(stage), self.dir / "d.nrrd", on_loss="error")


class TestCrossFormat(NrrdCase):
    def test_nifti_to_nrrd_keeps_world_positions_and_no_nifti_fields(self):
        """Review finding M18: NIfTI source fields never become NRRD custom keys."""
        for name, truth in (("plain", phantom.Truth()), ("time", phantom.Truth(shape=(5, 6, 7, 3)))):
            with self.subTest(case=name):
                source = phantom.write_nifti(self.dir / f"{name}.nii.gz", truth, extra={"descrip": "phantom 0.7"})
                image, _ = medvol.read(source)
                out = self.dir / f"{name}.nrrd"
                medvol.write(image, out)
                text = _header_text(out)
                self.assertNotIn(":=", text)
                self.assertNotIn("phantom 0.7", text)
                self.assertLessEqual(set(nrrd.read_header(str(out))), STANDARD_FIELDS)
                again, _ = medvol.read(out)
                self.assertLess(phantom.world_error(again, truth), 1e-4)
                np.testing.assert_array_equal(again.array, image.array)
                header = nrrd.read_header(str(out))
                if name == "time":
                    # the one 4-D space ITK reads correctly (KB ch04 P4)
                    self.assertEqual(header["space"], "left-posterior-superior-time")
                    self.assertEqual(header["kinds"], ["domain", "domain", "domain", "time"])
                    np.testing.assert_array_equal(header["space directions"][3], [0, 0, 0, 1.0])
                    self.assertEqual([(a.type, a.start, a.step, a.unit) for a in again.meta.axes[:1]],
                                     [("time", 0.0, 1.0, "second")])
                else:
                    self.assertEqual(header["space"], "left-posterior-superior")

    def test_ras_core_is_written_as_ras(self):
        truth = phantom.Truth()
        image, _ = medvol.read(phantom.write_nifti(self.dir / "a.nii.gz", truth), world_frame="ras")
        medvol.write(image, self.dir / "a.nrrd")
        self.assertEqual(nrrd.read_header(str(self.dir / "a.nrrd"))["space"], "right-anterior-superior")
        again, _ = medvol.read(self.dir / "a.nrrd")
        self.assertLess(phantom.world_error(again, truth), 1e-4)

    def test_nrrd_to_nifti_keeps_world_positions(self):
        truth = phantom.Truth()
        image, _ = medvol.read(phantom.write_nrrd(self.dir / "a.nrrd", truth, "left-anterior-superior"))
        medvol.write(image, self.dir / "a.nii.gz")
        again, _ = medvol.read(self.dir / "a.nii.gz")
        self.assertLess(phantom.world_error(again, truth), 1e-4)


if __name__ == "__main__":
    unittest.main()
