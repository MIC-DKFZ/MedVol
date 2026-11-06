"""Policy ``vector_frame`` (spec 5, 7.1, 7.2): the frame of vector components.

Files are written with nibabel and pynrrd, never with MedVol. Each test field holds, at every voxel, a known
world vector (LPS, mm); the stored components are that vector expressed in the file's own frame.
"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
from nibabel.openers import ImageOpener
from nibabel.volumeutils import array_to_file

import medvol
import phantom
from medvol.vectors import VectorFrameError

LPS = ["left", "posterior", "superior"]
RAS = ["right", "anterior", "superior"]
#: LPS -> coordinates of a NRRD space (a diagonal sign matrix)
SPACE_SIGNS = {"left-posterior-superior": np.eye(3), "right-anterior-superior": np.diag([-1.0, -1.0, 1.0]),
               "left-anterior-superior": np.diag([1.0, -1.0, 1.0])}
SHAPE = (5, 6, 7)                                  # file order i, j, k


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = np.deg2rad(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def world_field():
    """Known LPS vector per voxel, file order [i, j, k, 3]; all values exact in float32."""
    i, j, k = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in SHAPE], indexing="ij")
    return np.stack([i + 0.25, -2.0 * j + 1.5, 3.0 * k - 0.75], axis=-1)


def _as_mlarray_order(field):
    """[i, j, k, c] -> MLArray order [k, j, i, c]."""
    return np.ascontiguousarray(field.transpose(2, 1, 0, 3))


def write_nrrd(path, space="left-posterior-superior", frame=None, dtype="float32", kind="vector", field=None):
    """Vector NRRD (component axis fastest). ``frame`` is the spec's measurement-frame matrix (columns =
    components in space coordinates); None writes no measurement frame (components along the space axes)."""
    signs = SPACE_SIGNS[space]
    field = world_field() if field is None else field
    to_space = signs if frame is None else np.linalg.inv(frame) @ signs     # LPS -> measurement coordinates
    stored = np.einsum("rc,ijkc->ijkr", to_space, field).astype(dtype)
    linear = signs @ phantom.Truth().affine_lps[:3, :3]
    header = {
        "space": space,
        "space directions": np.vstack([np.full(3, np.nan), linear.T]),
        "space origin": signs @ phantom.Truth().affine_lps[:3, 3],
        "kinds": [kind, "domain", "domain", "domain"],
        "encoding": "raw",
    }
    if frame is not None:
        header["measurement frame"] = np.asarray(frame).T       # pynrrd: row r = file vector r = column r
    nrrd.write(str(path), np.moveaxis(stored, -1, 0), header)
    return path


def write_nifti(path, components, intent=1007, version=1):
    """Vector NIfTI as the standard lays it out: dim 4 = 1, components in dim 5 (``components``: [i, j, k, c])."""
    data = np.asarray(components)[:, :, :, None, :]
    header = (nib.Nifti1Header if version == 1 else nib.Nifti2Header)()
    header.set_data_dtype(data.dtype)
    header.set_data_shape(data.shape)
    header.set_xyzt_units("mm", "sec")
    affine = phantom.Truth().affine_ras
    header.set_qform(affine, code=1)
    header.set_sform(affine, code=1)
    header["intent_code"] = intent
    header["vox_offset"] = int(np.ceil(header.single_vox_offset / 16) * 16)
    with ImageOpener(str(path), "wb") as handle:
        header.write_to(handle)
        array_to_file(data, handle, data.dtype, offset=None, order="F")
    return path


def nrrd_header_text(path):
    return Path(path).read_bytes().split(b"\n\n", 1)[0].decode("ascii")


def components_lps(image):
    """The image's components re-expressed in LPS from its labels: array [k, j, i, 3]."""
    axis = next(i for i, a in enumerate(image.meta.axes) if a.type == "component")
    labels = image.meta.axes[axis].labels
    signs = {"left": (0, 1.0), "right": (0, -1.0), "posterior": (1, 1.0), "anterior": (1, -1.0),
             "superior": (2, 1.0), "inferior": (2, -1.0)}
    basis = np.zeros((3, 3))
    for column, word in enumerate(labels):
        row, sign = signs[word]
        basis[row, column] = sign
    values = np.moveaxis(np.asarray(image.array, dtype=np.float64), axis, -1)
    return values @ basis.T


class VectorCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def via_mla(self, path, out, **options):
        """X -> .mla -> Y (Y's format from ``out``); returns the write report."""
        image, _ = medvol.read(path, **options)
        mla = self.dir / (Path(out).name.split(".")[0] + ".mla")
        medvol.write(image, mla)
        again, _ = medvol.read(mla)
        self.assertEqual(again.meta.axes, image.meta.axes)          # .mla keeps the labels as they are
        return medvol.write(again, out)


class TestNiftiVectorIntent(VectorCase):
    def test_from_file_raises_and_names_the_options(self):
        for intent in (1006, 1007):
            with self.subTest(intent=intent):
                path = write_nifti(self.dir / f"v{intent}.nii", world_field().astype("float32"), intent=intent)
                with self.assertRaises(VectorFrameError) as caught:
                    medvol.read(path)
                self.assertIn("vector_frame='lps' or 'ras'", str(caught.exception))
                self.assertIn("does not declare", str(caught.exception))

    def test_error_raises(self):
        path = write_nifti(self.dir / "v.nii", world_field().astype("float32"))
        with self.assertRaises(VectorFrameError):
            medvol.read(path, vector_frame="error")

    def test_declared_frame_labels_the_components_without_changing_them(self):
        stored = world_field().astype("float32")
        path = write_nifti(self.dir / "v.nii", stored)
        for rule, words in (("lps", LPS), ("ras", RAS)):
            with self.subTest(vector_frame=rule):
                image, report = medvol.read(path, vector_frame=rule)
                self.assertEqual([(a.name, a.type) for a in image.meta.axes],
                                 [("c", "component"), ("z", "space"), ("y", "space"), ("x", "space")])
                self.assertEqual(list(image.meta.axes[0].labels), words)
                self.assertEqual(image.meta.values.kind, "vector")
                self.assertIn(rule, report.choices["vector_frame"])
                np.testing.assert_array_equal(image.array, stored.transpose(3, 2, 1, 0))
                self.assertEqual(list(image.meta.source[0].index_map["axis_order"]), [3, 2, 1, 0])

    def test_lps_declaration_gives_the_true_world_vectors(self):
        """An ITK/ANTs-style file (LPS components, intent 1007) read with vector_frame='lps'."""
        path = write_nifti(self.dir / "itk.nii", world_field().astype("float32"))
        image, _ = medvol.read(path, vector_frame="lps")
        np.testing.assert_array_equal(components_lps(image), _as_mlarray_order(world_field()))
        self.assertLess(np.max(np.abs(image.space().matrix()[:3] - phantom.Truth().affine_lps[:3])), 1e-4)

    def test_niftyreg_style_ras_file(self):
        """A NiftyReg-style file (RAS components, intent 1007) read with vector_frame='ras'."""
        field = world_field()
        ras = field * np.array([-1.0, -1.0, 1.0])
        image, _ = medvol.read(write_nifti(self.dir / "nr.nii", ras.astype("float32")), vector_frame="ras")
        np.testing.assert_array_equal(components_lps(image), _as_mlarray_order(field))

    def test_other_component_counts_need_no_frame(self):
        path = write_nifti(self.dir / "two.nii", np.zeros(SHAPE + (2,), dtype="float32"))
        image, report = medvol.read(path)
        self.assertIsNone(image.meta.axes[0].labels)
        self.assertIn("not applied", report.choices["vector_frame"])
        out = self.dir / "two.nrrd"
        report = medvol.write(image, out)
        self.assertTrue(any("frame of its components is unknown" in w for w in report.warnings))


class TestNrrdMeasurementFrame(VectorCase):
    def check_world(self, image):
        np.testing.assert_allclose(components_lps(image), _as_mlarray_order(world_field()), rtol=0, atol=1e-5)

    def test_without_measurement_frame_components_follow_the_space(self):
        for space, words in (("left-posterior-superior", LPS), ("right-anterior-superior", RAS),
                             ("left-anterior-superior", ["left", "anterior", "superior"])):
            with self.subTest(space=space):
                path = write_nrrd(self.dir / "v.nrrd", space)
                stored = nrrd.read(str(path), index_order="F")[0]
                image, report = medvol.read(path)
                self.assertEqual(list(image.meta.axes[-1].labels), words)
                self.assertEqual(image.meta.values.kind, "vector")
                np.testing.assert_array_equal(image.array, stored.T)          # values unchanged
                self.assertIn("no measurement frame", report.choices["vector_frame"])
                self.check_world(image)

    def test_identity_measurement_frame(self):
        path = write_nrrd(self.dir / "v.nrrd", "right-anterior-superior", frame=np.eye(3))
        image, report = medvol.read(path)
        self.assertEqual(list(image.meta.axes[-1].labels), RAS)
        np.testing.assert_array_equal(image.array, nrrd.read(str(path), index_order="F")[0].T)
        self.assertIn("values unchanged", report.choices["vector_frame"])
        self.check_world(image)

    def test_signed_permutation_measurement_frame(self):
        # component 0 along space y, component 1 along -space x, component 2 along -space z
        frame = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        for space in ("left-posterior-superior", "right-anterior-superior"):
            with self.subTest(space=space):
                path = write_nrrd(self.dir / "p.nrrd", space, frame=frame)
                image, report = medvol.read(path)
                expected = {"left-posterior-superior": ["posterior", "right", "inferior"],
                            "right-anterior-superior": ["anterior", "left", "inferior"]}[space]
                self.assertEqual(list(image.meta.axes[-1].labels), expected)
                np.testing.assert_array_equal(image.array, nrrd.read(str(path), index_order="F")[0].T)
                self.assertEqual(report.warnings, [])
                self.check_world(image)

    def test_rotated_measurement_frame_is_applied(self):
        frame = _rotation([1.0, 2.0, 0.5], 33.0)
        for space in ("left-posterior-superior", "right-anterior-superior"):
            for world_frame, words in (("dicom_patient", LPS), ("ras", RAS)):
                with self.subTest(space=space, world_frame=world_frame):
                    path = write_nrrd(self.dir / "r.nrrd", space, frame=frame, dtype="float64")
                    image, report = medvol.read(path, world_frame=world_frame)
                    self.assertEqual(list(image.meta.axes[-1].labels), words)
                    self.assertEqual(image.array.dtype, np.float64)
                    self.assertIn("applied", report.choices["vector_frame"])
                    self.assertTrue(any("values changed" in w for w in report.warnings))
                    np.testing.assert_allclose(components_lps(image), _as_mlarray_order(world_field()),
                                               rtol=0, atol=1e-12)

    def test_rotated_measurement_frame_float32(self):
        path = write_nrrd(self.dir / "r.nrrd", "left-posterior-superior", frame=_rotation([0, 0, 1], 20.0))
        image, _ = medvol.read(path)
        self.assertEqual(image.array.dtype, np.float32)
        self.check_world(image)

    def test_rotated_covariant_vectors(self):
        frame = _rotation([1.0, -1.0, 2.0], 50.0)
        image, _ = medvol.read(write_nrrd(self.dir / "g.nrrd", frame=frame, dtype="float64", kind="covariant-vector"))
        np.testing.assert_allclose(components_lps(image), _as_mlarray_order(world_field()), rtol=0, atol=1e-12)

    def test_rotated_measurement_frame_refuses_integer_data(self):
        path = write_nrrd(self.dir / "i.nrrd", frame=_rotation([0, 0, 1], 30.0), dtype="int16",
                          field=np.round(world_field()))
        with self.assertRaises(VectorFrameError) as caught:
            medvol.read(path)
        self.assertIn("int16", str(caught.exception))

    def test_policy_overrides_the_file(self):
        path = write_nrrd(self.dir / "v.nrrd", "right-anterior-superior", frame=np.eye(3))
        image, report = medvol.read(path, vector_frame="lps")
        self.assertEqual(list(image.meta.axes[-1].labels), LPS)
        np.testing.assert_array_equal(image.array, nrrd.read(str(path), index_order="F")[0].T)
        self.assertIn("declared by the policy", report.choices["vector_frame"])
        with self.assertRaises(VectorFrameError):
            medvol.read(path, vector_frame="error")
        # written back, the measurement frame states the declared frame in the file's RAS space
        out = self.dir / "out.nrrd"
        report = medvol.write(image, out)
        np.testing.assert_array_equal(nrrd.read_header(str(out))["measurement frame"], np.diag([-1.0, -1.0, 1.0]))
        self.assertTrue(any("measurement frame rewritten" in w for w in report.warnings))
        again, _ = medvol.read(out)
        self.assertEqual(list(again.meta.axes[-1].labels), LPS)


class TestRoundTrips(VectorCase):
    def test_nrrd_round_trip_is_exact(self):
        permutation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        cases = {
            "none_lps.nrrd": dict(),
            "none_ras.nrrd": dict(space="right-anterior-superior"),
            "identity.nrrd": dict(space="right-anterior-superior", frame=np.eye(3)),
            "perm.nrrd": dict(space="left-anterior-superior", frame=permutation),
        }
        for name, kwargs in cases.items():
            for world_frame in ("dicom_patient", "ras"):
                with self.subTest(file=name, world_frame=world_frame):
                    path = write_nrrd(self.dir / name, **kwargs)
                    out = self.dir / f"rt_{name}"
                    self.via_mla(path, out, world_frame=world_frame)
                    self.assertEqual(nrrd_header_text(out), nrrd_header_text(path))
                    self.assertEqual(Path(out).read_bytes(), Path(path).read_bytes())

    def test_nrrd_rotated_frame_round_trip_keeps_the_world_vectors(self):
        """The rotation is applied on read, so the rewritten file states the frame the values are now in."""
        path = write_nrrd(self.dir / "r.nrrd", "right-anterior-superior", frame=_rotation([1, 1, 1], 40.0),
                          dtype="float64")
        out = self.dir / "rt.nrrd"
        report = self.via_mla(path, out)
        self.assertTrue(any("measurement frame rewritten" in w for w in report.warnings))
        np.testing.assert_array_equal(nrrd.read_header(str(out))["measurement frame"], np.diag([-1.0, -1.0, 1.0]))
        again, _ = medvol.read(out)
        np.testing.assert_allclose(components_lps(again), _as_mlarray_order(world_field()), rtol=0, atol=1e-12)

    def test_nifti_round_trip_is_exact(self):
        for version in (1, 2):
            for rule in ("lps", "ras"):
                with self.subTest(version=version, vector_frame=rule):
                    path = write_nifti(self.dir / f"v{version}.nii", world_field().astype("float32"), version=version)
                    out = self.dir / f"rt{version}.nii"
                    report = self.via_mla(path, out, vector_frame=rule)
                    self.assertTrue(any(f"readers must assume {rule.upper()}" in w for w in report.warnings))
                    self.assertEqual(Path(out).read_bytes(), Path(path).read_bytes())

    def test_nrrd_to_nifti_to_nrrd(self):
        permutation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        cases = {   # space, measurement frame, labels on read, frame the NIfTI holds
            "lps": ("left-posterior-superior", None, LPS, "lps"),
            "ras": ("right-anterior-superior", None, RAS, "ras"),
            "perm": ("left-posterior-superior", permutation, ["posterior", "right", "superior"], "lps"),
        }
        for name, (space, frame, labels, held) in cases.items():
            with self.subTest(case=name):
                source = write_nrrd(self.dir / f"{name}.nrrd", space, frame=frame)
                image, _ = medvol.read(source)
                self.assertEqual(list(image.meta.axes[-1].labels), labels)
                nifti = self.dir / f"{name}.nii.gz"
                report = medvol.write(image, nifti)
                self.assertTrue(any(f"readers must assume {held.upper()}" in w for w in report.warnings))
                # a standard NIfTI vector file: dim 4 = 1, components in dim 5
                loaded = nib.load(str(nifti))
                self.assertEqual(loaded.shape, SHAPE + (1, 3))
                self.assertEqual(int(loaded.header["intent_code"]), 1007)
                written = np.asarray(loaded.dataobj)[:, :, :, 0, :]
                stored = np.moveaxis(nrrd.read(str(source), index_order="F")[0], 0, -1)
                if labels in (LPS, RAS):
                    self.assertIn("written as stored", report.choices["vector_frame"])
                    np.testing.assert_array_equal(written, stored)
                else:
                    self.assertIn("converted", report.choices["vector_frame"])
                    self.assertTrue(any("values changed" in w for w in report.warnings))
                    np.testing.assert_array_equal(written, world_field().astype("float32"))
                again, _ = medvol.read(nifti, vector_frame=held)
                out = self.dir / f"{name}_back.nrrd"
                medvol.write(again, out)
                final, _ = medvol.read(out)
                # NIfTI puts the components slowest: same axes, the component axis now first in array order
                self.assertEqual([(a.name, a.type) for a in final.meta.axes],
                                 [(a.name, a.type) for a in image.meta.axes[-1:] + image.meta.axes[:-1]])
                np.testing.assert_array_equal(components_lps(final), _as_mlarray_order(world_field()))
                self.assertLess(np.max(np.abs(final.space().matrix() - image.space().matrix())), 1e-4)

    def test_nifti_to_nrrd_writes_the_frame(self):
        ras = world_field() * np.array([-1.0, -1.0, 1.0])
        image, _ = medvol.read(write_nifti(self.dir / "nr.nii", ras.astype("float32")), vector_frame="ras")
        out = self.dir / "nr.nrrd"
        report = medvol.write(image, out)
        header = nrrd.read_header(str(out))
        self.assertEqual(header["space"], "left-posterior-superior")
        np.testing.assert_array_equal(header["measurement frame"], np.diag([-1.0, -1.0, 1.0]))
        self.assertIn("measurement frame written", report.choices["vector_frame"])
        # NIfTI holds the components slowest, so the NRRD gets them as its last axis
        self.assertEqual(header["kinds"], ["domain", "domain", "domain", "vector"])
        np.testing.assert_array_equal(nrrd.read(str(out), index_order="F")[0], ras.astype("float32"))
        again, _ = medvol.read(out)
        self.assertEqual(list(again.meta.axes[0].labels), RAS)
        np.testing.assert_array_equal(components_lps(again), _as_mlarray_order(world_field()))


if __name__ == "__main__":
    unittest.main()
