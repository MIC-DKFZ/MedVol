from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import nrrd
import pytest

from medvol import MedVol


REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_roundtrip_equal(first: MedVol, second: MedVol) -> None:
    assert np.array_equal(first.array, second.array)
    assert np.allclose(first.affine, second.affine)
    assert first.affine.shape == (first.array.ndim + 1, first.array.ndim + 1)
    assert second.affine.shape == (second.array.ndim + 1, second.array.ndim + 1)


def test_constructor_rejects_removed_arguments():
    with pytest.raises(TypeError):
        MedVol(np.zeros((3, 3)), is_seg=True)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        MedVol(np.zeros((3, 3)), copy=None)  # type: ignore[call-arg]



def test_affine_is_source_of_truth_and_geometry_is_snapped():
    affine = np.array(
        [
            [0.9999999, 0.0, 1.0999999],
            [0.0, 2.0000001, -3.0000001],
            [0.0, 0.0, 1.0],
        ]
    )
    image = MedVol(np.zeros((4, 5), dtype=np.float32), affine=affine)

    assert np.allclose(image.spacing, [1.0, 2.0])
    assert np.allclose(image.origin, [1.1, -3.0])
    assert np.allclose(image.direction, np.eye(2))
    assert np.allclose(image.translation, image.origin)
    assert np.allclose(image.scale, image.spacing)


def test_geometry_setters_rebuild_affine():
    image = MedVol(np.zeros((4, 5, 6), dtype=np.float32))
    image.spacing = [1.5, 2.0, 3.0]
    image.origin = [4.0, 5.0, 6.0]
    image.direction = [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    assert np.allclose(image.spacing, [1.5, 2.0, 3.0])
    assert np.allclose(image.origin, [4.0, 5.0, 6.0])
    assert np.allclose(
        image.affine[:3, :3],
        np.array(
            [
                [0.0, -2.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )


def test_invalid_backend_for_extension_raises(tmp_path):
    image = MedVol(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        image.save(tmp_path / "bad.nii.gz", backend="pynrrd")

    with pytest.raises(ValueError):
        MedVol(tmp_path / "bad.nrrd", backend="nibabel")


@pytest.mark.parametrize(
    ("shape", "suffix"),
    [
        ((4, 5), ".nii.gz"),
        ((4, 5, 6), ".nii.gz"),
        ((2, 4, 5, 6), ".nii.gz"),
        ((4, 5), ".nrrd"),
        ((4, 5, 6), ".nrrd"),
        ((2, 4, 5, 6), ".nrrd"),
    ],
)
def test_simpleitk_roundtrip(shape, suffix, tmp_path):
    affine = np.eye(len(shape) + 1, dtype=float)
    affine[:-1, :-1] = np.diag(np.arange(1, len(shape) + 1, dtype=float))
    affine[:-1, -1] = np.arange(10, 10 + len(shape), dtype=float)
    first = MedVol(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), affine=affine)

    path = tmp_path / f"simpleitk{suffix}"
    first.save(path, backend="simpleitk")
    second = MedVol(path, backend="simpleitk")

    assert_roundtrip_equal(first, second)
    assert isinstance(second.header, dict)


def test_simpleitk_rejects_incompatible_4d_nifti_affine(tmp_path):
    affine = np.eye(5, dtype=float)
    affine[0, 3] = 1.0
    image = MedVol(
        np.zeros((2, 3, 4, 5), dtype=np.float32),
        affine=affine,
    )

    with pytest.raises(ValueError):
        image.save(tmp_path / "bad_simpleitk_4d.nii.gz", backend="simpleitk")


@pytest.mark.parametrize("shape", [(4, 5), (4, 5, 6), (2, 4, 5, 6)])
def test_nibabel_roundtrip(shape, tmp_path):
    affine = np.eye(len(shape) + 1, dtype=float)
    if len(shape) == 2:
        affine[:2, :2] = [[2.0, 0.1], [0.0, 3.0]]
        affine[:2, 2] = [10.0, 11.0]
    elif len(shape) == 3:
        affine[:3, :3] = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        affine[:3, 3] = [10.0, 11.0, 12.0]
    else:
        affine[:3, :3] = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
        affine[:3, 4] = [10.0, 11.0, 12.0]
        affine[3, 3] = 5.0
        affine[3, 4] = 13.0

    first = MedVol(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), affine=affine)
    path = tmp_path / "nibabel.nii.gz"
    first.save(path, backend="nibabel")
    second = MedVol(path, backend="nibabel")

    assert_roundtrip_equal(first, second)
    assert isinstance(second.header, nib.nifti1.Nifti1Header)


def test_nibabel_rejects_coupled_4d_affine(tmp_path):
    affine = np.eye(5, dtype=float)
    affine[0, 3] = 1.0
    image = MedVol(np.zeros((2, 3, 4, 5), dtype=np.float32), affine=affine)

    with pytest.raises(ValueError):
        image.save(tmp_path / "coupled.nii.gz", backend="nibabel")


@pytest.mark.parametrize("shape", [(4, 5), (4, 5, 6), (2, 4, 5, 6)])
def test_pynrrd_roundtrip(shape, tmp_path):
    affine = np.eye(len(shape) + 1, dtype=float)
    affine[:-1, :-1] = np.diag(np.arange(1, len(shape) + 1, dtype=float))
    affine[:-1, -1] = np.arange(20, 20 + len(shape), dtype=float)
    first = MedVol(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), affine=affine)

    path = tmp_path / "pynrrd.nrrd"
    first.save(path, backend="pynrrd")
    second = MedVol(path, backend="pynrrd")

    assert_roundtrip_equal(first, second)
    assert isinstance(second.header, dict)


def test_pynrrd_loads_non_spatial_4d_axis(tmp_path):
    array = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape((2, 3, 4, 5))
    header = {
        "space": "left-posterior-superior-time",
        "space directions": np.array(
            [
                [np.nan, np.nan, np.nan],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
        "space origin": np.array([10.0, 11.0, 12.0]),
        "spacings": [7.0, 1.0, 2.0, 3.0],
    }
    path = tmp_path / "non_spatial.nrrd"
    nrrd.write(str(path), array, header=header)

    image = MedVol(path, backend="pynrrd")

    assert image.affine.shape == (5, 5)
    assert image.coordinate_system == "RAS+"
    assert image.array.shape == (3, 4, 5, 2)
    assert np.isclose(image.affine[3, 3], 7.0)


def test_backend_default_selection_and_raw_headers(tmp_path):
    nifti_source = MedVol(
        np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5)),
        affine=np.diag([2.0, 3.0, 4.0, 1.0]),
    )
    nifti_path = tmp_path / "default_source.nii.gz"
    nifti_source.save(nifti_path)

    nifti = MedVol(nifti_path)
    assert nifti.backend == "nibabel"
    assert isinstance(nifti.header, nib.nifti1.Nifti1Header)

    nrrd_source = MedVol(
        np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape((2, 3, 4, 5)),
        affine=np.diag([1.0, 2.0, 3.0, 4.0, 1.0]),
    )
    nrrd_path = tmp_path / "default_source.nrrd"
    nrrd_source.save(nrrd_path)

    nrrd_image = MedVol(nrrd_path)
    assert nrrd_image.backend == "pynrrd"
    assert isinstance(nrrd_image.header, dict)

    path = tmp_path / "cross_backend.nrrd"
    nifti.save(path)
    reloaded = MedVol(path)
    assert reloaded.backend == "pynrrd"


def test_coordinate_system_is_backend_independent_by_default():
    path = REPO_ROOT / "examples/data/3d_img.nii.gz"
    nib_image = MedVol(path, backend="nibabel")
    sitk_image = MedVol(path, backend="simpleitk")

    assert nib_image.coordinate_system is not None
    assert sitk_image.coordinate_system is not None
    assert nib_image.coordinate_system == "RAS+"
    assert sitk_image.coordinate_system == "RAS+"
    assert np.array_equal(nib_image.array, sitk_image.array)
    assert np.allclose(nib_image.affine, sitk_image.affine)


def test_canonicalize_false_preserves_backend_native_differences():
    path = REPO_ROOT / "examples/data/3d_img.nii.gz"
    nib_image = MedVol(path, backend="nibabel", canonicalize=False)
    sitk_image = MedVol(path, backend="simpleitk", canonicalize=False)

    assert nib_image.coordinate_system != sitk_image.coordinate_system
    assert not np.array_equal(nib_image.array, sitk_image.array)


def test_coordinate_system_param_lps_canonicalizes_to_ras():
    # An LPS+ array with identity direction: axes 0 and 1 must be flipped to RAS+.
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([1.5, 2.0, 3.0, 1.0])

    image = MedVol(array, affine=affine, coordinate_system="LPS+")

    assert image.coordinate_system == "RAS+"
    # Axes 0 and 1 should be flipped relative to the input.
    assert np.array_equal(image.array, np.flip(np.flip(array, axis=0), axis=1))


def test_coordinate_system_param_lps_no_canonicalize_preserves_orientation():
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([1.5, 2.0, 3.0, 1.0])

    image = MedVol(array, affine=affine, coordinate_system="LPS+", canonicalize=False)

    assert image.coordinate_system == "LPS+"
    assert np.array_equal(image.array, array)


def test_coordinate_system_param_ras_is_noop():
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([1.5, 2.0, 3.0, 1.0])

    default_image = MedVol(array, affine=affine)
    explicit_image = MedVol(array, affine=affine, coordinate_system="RAS+")

    assert np.array_equal(default_image.array, explicit_image.array)
    assert np.allclose(default_image.affine, explicit_image.affine)
    assert explicit_image.coordinate_system == "RAS+"


def test_coordinate_system_param_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported coordinate_system"):
        MedVol(np.zeros((3, 4, 5), dtype=np.float32), coordinate_system="scanner-xyz")


def test_coordinate_system_param_with_filepath_raises(tmp_path):
    image = MedVol(np.zeros((3, 4, 5), dtype=np.float32))
    path = tmp_path / "img.nii.gz"
    image.save(path)

    with pytest.raises(ValueError, match="coordinate_system cannot be set when loading from a file"):
        MedVol(path, coordinate_system="RAS+")


def test_get_geometry_identity():
    """get_geometry("RAS+") returns the same geometry as the internal state."""
    affine = np.diag([1.5, 2.0, 3.0, 1.0])
    affine[:3, 3] = [10.0, 11.0, 12.0]
    image = MedVol(np.zeros((4, 5, 6), dtype=np.float32), affine=affine)

    geom = image.get_geometry("RAS+")

    assert geom["coordinate_system"] == "RAS+"
    assert np.allclose(geom["affine"], image.affine)
    assert np.allclose(geom["spacing"], image.spacing)
    assert np.allclose(geom["origin"], image.origin)
    assert not geom["oblique"]


def test_get_geometry_lps():
    """get_geometry("LPS+") flips axes 0 and 1 and negates the corresponding world rows."""
    sx, sy, sz = 1.5, 2.0, 3.0
    ox, oy, oz = 10.0, 11.0, 12.0
    Nx, Ny, Nz = 4, 5, 6
    affine = np.diag([sx, sy, sz, 1.0])
    affine[:3, 3] = [ox, oy, oz]
    image = MedVol(np.zeros((Nx, Ny, Nz), dtype=np.float32), affine=affine)

    geom = image.get_geometry("LPS+")

    assert geom["coordinate_system"] == "LPS+"
    # Spacing is always positive.
    assert np.allclose(geom["spacing"][:3], [sx, sy, sz])
    # Origin shifts to the far corners of the flipped axes.
    expected_origin = [-(ox + sx * (Nx - 1)), -(oy + sy * (Ny - 1)), oz]
    assert np.allclose(geom["origin"][:3], expected_origin)
    assert not geom["oblique"]
    # Internal state untouched.
    assert image.coordinate_system == "RAS+"
    assert np.allclose(image.origin, [ox, oy, oz])


def test_get_geometry_permuted_axes():
    """get_geometry("ASR+") reorders axes A→0, S→1, R→2."""
    sx, sy, sz = 1.5, 2.0, 3.0
    ox, oy, oz = 10.0, 11.0, 12.0
    affine = np.diag([sx, sy, sz, 1.0])
    affine[:3, 3] = [ox, oy, oz]
    image = MedVol(np.zeros((4, 5, 6), dtype=np.float32), affine=affine)

    geom = image.get_geometry("ASR+")

    # ASR+ → axis 0 = A (sy), axis 1 = S (sz), axis 2 = R (sx).
    assert np.allclose(geom["spacing"][:3], [sy, sz, sx])
    assert np.allclose(geom["origin"][:3], [oy, oz, ox])
    assert not geom["oblique"]


def test_get_geometry_deoblique():
    """get_geometry(deoblique=True) returns a diagonal affine with same origin."""
    affine = np.array(
        [
            [1.0, 0.2, 0.0, 10.0],
            [0.0, 2.0, 0.3, 11.0],
            [0.0, 0.0, 3.0, 12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    image = MedVol(np.zeros((4, 5, 6), dtype=np.float32), affine=affine)

    geom = image.get_geometry("RAS+", deoblique=True)

    assert np.allclose(geom["affine"][:3, :3], np.diag(geom["spacing"][:3]))
    assert np.allclose(geom["origin"][:3], [10.0, 11.0, 12.0])
    # Internal affine is NOT deobliqued.
    assert not np.allclose(image.affine[:3, :3], np.diag(image.spacing))


def test_get_geometry_oblique_flag():
    """oblique=True for non-axis-aligned direction, False otherwise."""
    diagonal_affine = np.diag([1.5, 2.0, 3.0, 1.0])
    oblique_affine = np.array(
        [[1.0, 0.2, 0.0, 0.0], [0.0, 2.0, 0.3, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    non_oblique = MedVol(np.zeros((3, 4, 5), dtype=np.float32), affine=diagonal_affine)
    oblique = MedVol(np.zeros((3, 4, 5), dtype=np.float32), affine=oblique_affine)

    assert not non_oblique.get_geometry("RAS+")["oblique"]
    assert oblique.get_geometry("RAS+")["oblique"]


def test_get_geometry_requires_canonicalize():
    image = MedVol(
        np.zeros((3, 4, 5), dtype=np.float32),
        coordinate_system="RAS+",
        canonicalize=False,
    )
    with pytest.raises(ValueError, match="canonicalize"):
        image.get_geometry("LPS+")


def test_get_array_identity():
    """get_array("RAS+") returns the same data as the internal array."""
    image = MedVol(np.arange(60, dtype=np.float32).reshape(3, 4, 5))

    result = image.get_array("RAS+")

    assert np.array_equal(result, image.array)


def test_get_array_lps():
    """get_array("LPS+") flips axes 0 and 1."""
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    image = MedVol(array)

    result = image.get_array("LPS+")

    expected = np.flip(np.flip(image.array, axis=0), axis=1)
    assert np.array_equal(result, expected)
    # Zero-copy view — no data duplication.
    assert result.base is not None


def test_get_array_permuted():
    """get_array("ASR+") transposes to (A, S, R) order."""
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    image = MedVol(array)

    result = image.get_array("ASR+")

    assert result.shape == (image.array.shape[1], image.array.shape[2], image.array.shape[0])
    assert np.array_equal(result, np.transpose(image.array, (1, 2, 0)))


def test_get_array_requires_canonicalize():
    image = MedVol(
        np.zeros((3, 4, 5), dtype=np.float32),
        coordinate_system="RAS+",
        canonicalize=False,
    )
    with pytest.raises(ValueError, match="canonicalize"):
        image.get_array("LPS+")
