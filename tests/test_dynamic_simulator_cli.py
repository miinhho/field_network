from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DynamicSimulatorCLITests(unittest.TestCase):
    def test_cli_table_output(self) -> None:
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "ffrag.dynamic_simulator_cli",
            "--steps",
            "2",
            "--format",
            "table",
        ]
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("step  objective", out.stdout)

    def test_cli_json_output(self) -> None:
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "ffrag.dynamic_simulator_cli",
            "--steps",
            "2",
            "--format",
            "json",
        ]
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("\"frames\"", out.stdout)


if __name__ == "__main__":
    unittest.main()
