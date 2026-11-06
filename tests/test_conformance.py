"""Every reader x writer pair keeps the world position of every voxel (spec 9.1)."""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import medvol
import phantom
from medvol import registry

warnings.filterwarnings("ignore", module="pydicom")


def _dicom(path, truth):
    phantom.write_dicom_series(path, truth)
    return path


# source format -> (file name, writer); truth dtype int16 fits every format
SOURCES = {
    "nifti": ("a.nii.gz", phantom.write_nifti),
    "mgh": ("a.mgz", phantom.write_mgh),
    "dicom": ("dicom_in", _dicom),
}
TARGETS = {"mla": "b.mla", "nifti": "b.nii.gz", "mgh": "b.mgz", "nrrd": "b.nrrd", "metaimage": "b.mha"}


def _optional_sources():
    sources = dict(SOURCES)
    try:
        import phantom_metaimage

        writer = getattr(phantom_metaimage, "write_mha", None) or getattr(phantom_metaimage, "write_metaimage", None)
        if writer is not None:
            sources["metaimage"] = ("a.mha", lambda path, truth: writer(path, truth))
    except ImportError:
        pass
    sources["nrrd"] = ("a.nrrd", phantom.write_nrrd)
    return sources


class TestConformance(unittest.TestCase):
    def test_every_reader_writer_pair_keeps_world_positions(self):
        available = registry.plugins()
        for source, (name, write) in _optional_sources().items():
            if source not in available:
                continue
            for target, out_name in TARGETS.items():
                if target not in available:
                    continue
                for sheared in (False, True):
                    with self.subTest(source=source, target=target, sheared=sheared), tempfile.TemporaryDirectory() as tmp:
                        truth = phantom.sheared_truth() if sheared else phantom.Truth()
                        truth.dtype = "int16"
                        image, _ = medvol.read(write(Path(tmp) / name, truth))
                        self.assertLess(phantom.world_error(image, truth), 1e-4)
                        if sheared and not available[target].capabilities.shear:
                            with self.assertRaises(medvol.LossError):
                                medvol.write(image, Path(tmp) / out_name, on_loss="error")
                            continue
                        medvol.write(image, Path(tmp) / out_name)
                        again, _ = medvol.read(Path(tmp) / out_name)
                        self.assertLess(phantom.world_error(again, truth), 1e-4)
                        values = image.array.astype(np.float64).reshape(-1)
                        self.assertTrue(np.array_equal(np.sort(again.array.astype(np.float64).reshape(-1)), np.sort(values)))


if __name__ == "__main__":
    unittest.main()
