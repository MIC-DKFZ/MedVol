import tempfile
import unittest
from pathlib import Path

import numpy as np
from mlarray import MLArray

import medvol
from medvol import registry
from medvol.convert import check_capabilities
from medvol.image import Image
from medvol.plugins.base import Capabilities
from medvol.policy import Policy
from medvol.report import LossError, Report


def _mla(path, **kwargs):
    MLArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4), spacing=(1, 2, 3), space="patient", **kwargs).save(path)


class TestPolicy(unittest.TestCase):
    def test_defaults_follow_the_spec(self):
        policy = Policy()
        self.assertEqual((policy.nifti_form, policy.nifti_no_codes, policy.reorient, policy.world_frame),
                         ("nibabel", "error", "none", "dicom_patient"))
        self.assertEqual((policy.dicom_tilt, policy.dicom_irregular_spacing, policy.dicom_stacks), ("exact", "error", "split"))
        self.assertEqual((policy.vector_frame, policy.on_loss, policy.layout), ("from_file", "warn", "source"))

    def test_unknown_options_and_values_are_refused(self):
        with self.assertRaises(ValueError):
            Policy(world_frame="lps")
        with self.assertRaises(TypeError):
            Policy.from_options(colour="red")
        self.assertEqual(Policy.from_options(Policy(on_loss="error"), world_frame="ras").on_loss, "error")


class TestRegistryAndReport(unittest.TestCase):
    def test_mla_is_found_by_content_and_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.mla"
            _mla(path)
            self.assertEqual(registry.for_reading(path).name, "mla")
            self.assertEqual(registry.for_writing("x.mla").name, "mla")
            with self.assertRaises(ValueError):
                registry.for_writing("x.unknown")

    def test_report_records_and_raises_on_loss(self):
        report = Report("write", "nifti")
        report.choose("nifti_form", "sform")
        report.drop("user", "no place for it")
        with self.assertRaises(LossError):
            report.drop("user", "no place for it", on_loss="error")
        entry = report.history_entry("medvol 2")
        self.assertEqual(entry["settings"], {"nifti_form": "sform"})
        self.assertEqual(entry["warnings"], ["dropped user: no place for it"])


class TestCapabilities(unittest.TestCase):
    def _image(self, **kwargs):
        return Image(np.zeros((2, 3, 4), dtype=np.float32), MLArray(np.zeros((2, 3, 4)), **kwargs).meta)

    def test_unrepresentable_metadata_is_reported_or_refused(self):
        image = self._image(values={"kind": "label", "labels": {"1": {"name": "liver"}}}, user={"a": 1})
        report = Report("write", "x")
        check_capabilities(image, Capabilities(), report, Policy())
        self.assertEqual(len(report.dropped), 2)
        with self.assertRaises(LossError):
            check_capabilities(image, Capabilities(), Report("write", "x"), Policy(on_loss="error"))

    def test_axes_the_format_cannot_hold_always_raise(self):
        image = Image(np.zeros((2, 3, 4)), MLArray(np.zeros((2, 3, 4)), axes=["z", "y", {"name": "c", "type": "channel"}]).meta)
        with self.assertRaises(LossError):
            check_capabilities(image, Capabilities(), Report("write", "x"), Policy())

    def test_shear_is_reported_for_formats_without_it(self):
        matrix = [[1.0, 0.3, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        spaces = [{"name": "patient", "axes": [{"name": a, "unit": "millimeter"} for a in "xyz"],
                   "from_array": {"array_axes": ["axis_0", "axis_1", "axis_2"], "matrix": matrix}}]
        report = Report("write", "x")
        check_capabilities(self._image(spaces=spaces), Capabilities(shear=False), report, Policy())
        self.assertIn("sheared", report.dropped[0])


class TestMla(unittest.TestCase):
    def test_round_trip_keeps_everything(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, target = Path(tmpdir) / "a.mla", Path(tmpdir) / "b.mla"
            _mla(source, user={"a": 1}, source=[{"format": "nifti", "header": {"descrip": "x"}}])
            report = medvol.convert(source, target)
            self.assertEqual(report.dropped, [])
            a, b = MLArray(source), MLArray(target)
            self.assertTrue(np.array_equal(a.to_numpy(), b.to_numpy()))
            self.assertEqual(a.meta.to_dict(include_lazy=True), b.meta.to_dict(include_lazy=True))

    def test_inspect_summarises_in_mlarray_terms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.mla"
            _mla(path)
            text = medvol.inspect(path)
            self.assertIn("space patient", text)
            self.assertIn("letters LPS", text)


if __name__ == "__main__":
    unittest.main()
