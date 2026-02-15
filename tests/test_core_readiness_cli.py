from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CoreReadinessCLITests(unittest.TestCase):
    def test_cli_runs_and_prints_summary_header(self) -> None:
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "ffrag.core_readiness_cli",
            "--modules",
            "tests.test_core_readiness",
        ]
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("module_count,tests_run,failures,errors,successful", out.stdout)


if __name__ == "__main__":
    unittest.main()
