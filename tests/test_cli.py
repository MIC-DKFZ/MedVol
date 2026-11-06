"""The ``medvol`` command line (spec section 8)."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import medvol
import phantom
from medvol.cli import main


def _run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(list(argv))
    return out.getvalue()


class TestCli(unittest.TestCase):
    def test_convert_inspect_and_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = phantom.write_nifti(Path(tmp) / "a.nii.gz")
            report = _run("convert", str(source), str(Path(tmp) / "b.nrrd"), "--world-frame", "ras", "--on-loss", "error")
            self.assertIn("convert nifti -> nrrd", report)
            image, _ = medvol.read(Path(tmp) / "b.nrrd")
            self.assertLess(phantom.world_error(image, phantom.Truth()), 1e-4)
            summary = _run("inspect", str(Path(tmp) / "b.nrrd"))
            self.assertIn("1 image(s)", summary)
            self.assertIn("space patient", summary)
            self.assertIn("nifti", _run("formats"))

    def test_bad_input_exits_with_an_error(self):
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            main(["inspect", "/nonexistent/file.xyz"])
        self.assertEqual(raised.exception.code, 1)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main(["convert", "a", "b", "--nifti-form", "bogus"])

    def test_console_script_entry_point(self):
        result = subprocess.run([sys.executable, "-m", "medvol.cli", "--version"], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), f"medvol {medvol.__version__}")


if __name__ == "__main__":
    unittest.main()
