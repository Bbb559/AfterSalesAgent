from __future__ import annotations

import unittest
from pathlib import Path


class ProjectStructureTest(unittest.TestCase):
    def test_project_uses_single_backend_config_file(self) -> None:
        root = Path(__file__).resolve().parents[1]

        self.assertTrue((root / "backend" / "config.py").exists())
        self.assertFalse((root / "backend" / "rag" / "config.py").exists())


if __name__ == "__main__":
    unittest.main()
