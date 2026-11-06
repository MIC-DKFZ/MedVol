"""Command line: ``medvol convert`` and ``medvol inspect`` (spec section 8)."""

import argparse
from typing import List, Optional

from medvol.policy import OPTIONS, Policy


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("policies", "How ambiguities are resolved (the first choice is the default).")
    for name, choices in OPTIONS.items():
        group.add_argument(f"--{name.replace('_', '-')}", choices=choices, help=f"default: {choices[0]}")
    group.add_argument("--nifti-form-tolerance-mm", type=float,
                       help=f"qform/sform disagreement tolerance in mm (default: {Policy().nifti_form_tolerance_mm})")


def _policy_options(args: argparse.Namespace) -> dict:
    names = list(OPTIONS) + ["nifti_form_tolerance_mm"]
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medvol",
        description="Faithful conversion of medical image formats (NIfTI, NRRD, MetaImage, MGH/MGZ, DICOM, .mla). "
                    "Nothing is reoriented, rescaled or dropped silently; the report lists every choice.",
    )
    from medvol import __version__
    parser.add_argument("--version", action="version", version=f"medvol {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="Convert INPUT to OUTPUT and print the report.")
    convert.add_argument("input", help="Input file or DICOM directory.")
    convert.add_argument("output", help="Output file (or directory for DICOM).")
    convert.add_argument("--source-format", help="Input format if it cannot be detected (e.g. dicom).")
    convert.add_argument("--target-format", help="Output format if the extension does not say (e.g. dicom).")
    _add_policy_arguments(convert)

    inspect = commands.add_parser("inspect", help="Print axes, spaces, values and warnings of INPUT.")
    inspect.add_argument("input", help="Input file or DICOM directory.")
    inspect.add_argument("--format", help="Input format if it cannot be detected (e.g. dicom).")
    _add_policy_arguments(inspect)

    commands.add_parser("formats", help="List the available format plugins.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    import medvol
    from medvol import registry

    try:
        if args.command == "convert":
            print(medvol.convert(args.input, args.output, source_format=args.source_format,
                                 target_format=args.target_format, **_policy_options(args)))
        elif args.command == "inspect":
            print(medvol.inspect(args.input, format=args.format, **_policy_options(args)))
        else:
            for name, plugin in sorted(registry.plugins().items()):
                print(f"{name:<10} {' '.join(plugin.extensions)}")
    except (OSError, ValueError, RuntimeError, NotImplementedError) as error:
        parser.exit(1, f"medvol: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
