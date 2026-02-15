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
            "--nodes",
            "120",
            "--format",
            "json",
        ]
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("\"frames\"", out.stdout)
        self.assertIn("\"node_positions\"", out.stdout)

    def test_cli_html_output(self) -> None:
        out_path = ROOT / "tests" / "tmp_simulator_trace.html"
        if out_path.exists():
            out_path.unlink()
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "ffrag.dynamic_simulator_cli",
            "--steps",
            "2",
            "--format",
            "html",
            "--out",
            str(out_path),
        ]
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("wrote_html", out.stdout)
        self.assertTrue(out_path.exists())
        text = out_path.read_text(encoding="utf-8")
        self.assertIn("<html", text.lower())
        out_path.unlink()

    def test_cli_webgl_output(self) -> None:
        out_path = ROOT / "tests" / "tmp_simulator_trace_webgl.html"
        if out_path.exists():
            out_path.unlink()
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "ffrag.dynamic_simulator_cli",
            "--steps",
            "2",
            "--nodes",
            "300",
            "--position-model",
            "physics",
            "--format",
            "webgl",
            "--out",
            str(out_path),
        ]
        out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertIn("wrote_webgl", out.stdout)
        self.assertTrue(out_path.exists())
        text = out_path.read_text(encoding="utf-8")
        self.assertIn("getContext(\"webgl\"", text)
        out_path.unlink()


if __name__ == "__main__":
    unittest.main()
