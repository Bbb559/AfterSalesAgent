from __future__ import annotations

import unittest
from pathlib import Path


class ProjectStructureTest(unittest.TestCase):
    def test_project_uses_single_backend_config_file(self) -> None:
        root = Path(__file__).resolve().parents[1]

        self.assertTrue((root / "backend" / "config.py").exists())
        self.assertFalse((root / "backend" / "rag" / "config.py").exists())

    def test_warranty_rule_file_was_removed_after_tool_cleanup(self) -> None:
        root = Path(__file__).resolve().parents[1]

        self.assertFalse((root / "backend" / "rules" / "warranty_rules.py").exists())

    def test_old_escalation_rule_and_tool_files_were_removed(self) -> None:
        root = Path(__file__).resolve().parents[1]

        self.assertFalse((root / "backend" / "rules" / "escalation_rules.py").exists())
        self.assertFalse((root / "backend" / "tools" / "escalation.py").exists())
        self.assertFalse((root / "backend" / "tools" / "safety.py").exists())
        self.assertFalse((root / "backend" / "tools" / "ticket.py").exists())
        self.assertFalse((root / "backend" / "tools" / "local_runner.py").exists())
        self.assertTrue((root / "backend" / "tools" / "warranty.py").exists())
        self.assertTrue((root / "backend" / "rules" / "dispatch_rules.py").exists())

    def test_agent_rule_boundary_is_explicit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        agents_dir = root / "backend" / "agents"

        self.assertFalse((agents_dir / "text_guards.py").exists())
        self.assertFalse((root / "backend" / "rules" / "risk_rules.py").exists())
        self.assertTrue((root / "backend" / "rules" / "safety_rules.py").exists())
        self.assertTrue((root / "backend" / "rules" / "output_rules.py").exists())
        self.assertTrue((root / "backend" / "rules" / "case_rules.py").exists())

        forbidden_tokens = [
            "backend.rules",
            "text_guards",
            "FREE_PROMISES",
            "DANGEROUS_ACTIONS",
            "_apply_reply_guards",
            "_apply_safety_guard",
        ]
        for path in agents_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, text, path.name)

    def test_tools_do_not_contain_rules_only_wrappers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tools_dir = root / "backend" / "tools"

        forbidden_tokens = [
            "ChargerSafetyTool",
            "ChargerDispatchTool",
            "LocalToolRunner",
            "charger_safety_guard",
            "charger_dispatch_draft",
        ]
        for path in tools_dir.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, text, path.name)


if __name__ == "__main__":
    unittest.main()
