
import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestJSONLogging(unittest.TestCase):

    def test_prediction_json_output(self):
        root = Path(__file__).resolve().parents[1]

        command = (
            "import runpy; "
            "runpy.run_path("
            "'src/mess_waste/json_logging.py'"
            ")['log_prediction'](12.5)"
        )

        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=root,
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 1)

        log = json.loads(lines[0])

        self.assertEqual(log["level"], "INFO")
        self.assertEqual(log["event"], "prediction")
        self.assertEqual(log["prediction_kg"], 12.5)
        self.assertIn("timestamp", log)
