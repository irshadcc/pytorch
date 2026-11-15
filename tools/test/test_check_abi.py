import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.abi import check_abi


BASE = "struct point { int x; }; int torch_test(struct point *p) { return p->x; }"


@instantiate_parametrized_tests
class TestCheckABI(TestCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def make_snapshot(self, name, source, debug=True):
        directory = self.directory / name
        directory.mkdir()
        code = directory / "source.c"
        code.write_text(source)
        command = [
            "gcc",
            "-shared",
            "-fPIC",
            str(code),
            "-o",
            str(directory / "libtorch_cpu.so"),
        ]
        if debug:
            command.extend(["-g", "-gdwarf-4"])
        subprocess.run(command, check=True, capture_output=True)
        args = argparse.Namespace(
            build=directory,
            output=directory / "abi",
            libraries=["libtorch_cpu.so"],
            configuration="test",
        )
        check_abi.snapshot(args)
        return args.output

    @unittest.skipUnless(
        shutil.which("gcc") and shutil.which("abidw") and shutil.which("abidiff"),
        "requires gcc and libabigail",
    )
    @parametrize(
        "source, expected",
        [
            (BASE, 0),
            (BASE + " int torch_added(void) { return 0; }", 0),
            (BASE + " int unrelated(void) { return 0; } int unrelated_var = 0;", 0),
            (BASE.replace("int x;", "long x;"), 1),
            (BASE.replace("struct point *p)", "struct point *p, double factor)"), 1),
            ("int torch_other(void) { return 0; }", 1),
        ],
    )
    def test_compiled_library(self, source, expected):
        baseline = self.make_snapshot("baseline", BASE)
        current = self.make_snapshot("current", source)
        args = argparse.Namespace(
            baseline=baseline,
            current=current,
            report=self.directory / "report.md",
        )
        self.assertEqual(check_abi.compare(args), expected)
        self.assertIn("libtorch_cpu.so", args.report.read_text())
        self.assertNotIn("unrelated", (current / "libtorch_cpu.so.abi").read_text())

    @unittest.skipUnless(
        shutil.which("gcc") and shutil.which("abidw"),
        "requires gcc and libabigail",
    )
    def test_missing_debug_info(self):
        with self.assertRaisesRegex(
            RuntimeError, "missing stable ABI type information"
        ):
            self.make_snapshot("stripped", BASE, debug=False)

    @parametrize(
        "code, expected",
        [(0, 0), (4, 1), (8, 1), (12, 1), (1, 2), (2, 2), (5, 2), (-9, 2)],
    )
    def test_exit_bits(self, code, expected):
        corpus = """<abi-corpus><elf-function-symbols>
<elf-symbol name='torch_test'/></elf-function-symbols>
<abi-instr><function-decl name='torch_test'/></abi-instr></abi-corpus>"""
        metadata = json.dumps({"libraries": ["libtorch_cpu.so"]})
        (self.directory / "metadata.json").write_text(metadata)
        (self.directory / "libtorch_cpu.so.abi").write_text(corpus)
        args = argparse.Namespace(
            baseline=self.directory,
            current=self.directory,
            report=self.directory / "report.md",
        )
        result = subprocess.CompletedProcess([], code, stdout="fixture")
        with mock.patch.object(check_abi, "run", return_value=result):
            self.assertEqual(check_abi.compare(args), expected)


if __name__ == "__main__":
    run_tests()
