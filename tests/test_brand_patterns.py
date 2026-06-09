"""测试 data/rules/brand_patterns.json 加载与 case_rules 兜底逻辑。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.rules import case_rules


class BrandPatternsTest(unittest.TestCase):
    """验证品牌/型号配置化加载和硬编码兜底行为不变。"""

    def setUp(self) -> None:
        """重置模块级缓存，确保每个测试独立。"""
        case_rules._BRAND_PATTERNS_CACHE = None
        self._temp_dir = tempfile.TemporaryDirectory()
        self._temp_rules = Path(self._temp_dir.name) / "brand_patterns.json"

    def tearDown(self) -> None:
        case_rules._BRAND_PATTERNS_CACHE = None
        self._temp_dir.cleanup()

    def test_json_missing_returns_empty_config(self) -> None:
        """JSON 文件不存在时 _load_brand_patterns 返回空字典。"""
        with patch.object(case_rules, "_BRAND_PATTERNS_PATH", self._temp_rules):
            self._temp_rules.unlink(missing_ok=True)
            result = case_rules._load_brand_patterns()
            self.assertEqual(result, {})

    def test_json_present_loads_config(self) -> None:
        """JSON 文件存在时正确加载配置。"""
        config = {
            "brands": [
                {
                    "name": "VoltGate",
                    "aliases": ["VoltGate", "VG"],
                    "model_patterns": [r"\b(VG-[A-Z0-9]+(?:-[A-Z0-9]+)*)\b"],
                    "default": True,
                }
            ]
        }
        self._temp_rules.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        with patch.object(case_rules, "_BRAND_PATTERNS_PATH", self._temp_rules):
            case_rules._BRAND_PATTERNS_CACHE = None
            result = case_rules._load_brand_patterns()
            self.assertIn("brands", result)
            self.assertEqual(result["brands"][0]["name"], "VoltGate")

    def test_build_model_regexes_from_config(self) -> None:
        """从配置中正确提取所有品牌的型号正则模式。"""
        config = {
            "brands": [
                {
                    "name": "VoltGate",
                    "model_patterns": [r"\bVG-\w+", r"\bVGPro-\w+"],
                },
                {
                    "name": "Huawei",
                    "model_patterns": [r"\bHW-\w+"],
                },
            ]
        }
        patterns = case_rules._build_model_regexes_from_config(config)
        self.assertIn(r"\bVG-\w+", patterns)
        self.assertIn(r"\bVGPro-\w+", patterns)
        self.assertIn(r"\bHW-\w+", patterns)
        self.assertEqual(len(patterns), 3)

    def test_build_model_regexes_from_empty_config_returns_empty(self) -> None:
        """空配置返回空列表。"""
        self.assertEqual(case_rules._build_model_regexes_from_config({}), [])

    def test_build_brand_names_from_config(self) -> None:
        """从配置中提取所有品牌名称和别名。"""
        config = {
            "brands": [
                {"name": "VoltGate", "aliases": ["VG", "voltgate"]},
                {"name": "华为", "aliases": []},
            ]
        }
        names = case_rules._build_brand_names_from_config(config)
        self.assertIn("VoltGate", names)
        self.assertIn("VG", names)
        self.assertIn("voltgate", names)
        self.assertIn("华为", names)

    def test_structured_fallback_hardcoded_fallback_still_works(self) -> None:
        """JSON 缺失时硬编码 VoltGate 兜底行为不变。"""
        with patch.object(case_rules, "_BRAND_PATTERNS_PATH", self._temp_rules):
            self._temp_rules.unlink(missing_ok=True)
            case_rules._BRAND_PATTERNS_CACHE = None
            result = case_rules._structured_fallback("VoltGate VG-11KW-Pro 屏幕 VG-E01")
            self.assertEqual(result["brand"], "VoltGate")
            self.assertEqual(result["charger_model"], "VG-11KW-Pro")

    def test_structured_fallback_with_json_config(self) -> None:
        """JSON 配置存在时从配置中识别品牌。"""
        config = {
            "brands": [
                {
                    "name": "VoltGate",
                    "aliases": ["VoltGate", "VG"],
                    "model_patterns": [
                        r"\b(VG-[A-Z0-9]+(?:-[A-Z0-9]+)*)\b",
                        r"\b(VG-[A-Za-z]+[A-Za-z0-9]*)\b",
                    ],
                    "default": True,
                }
            ]
        }
        self._temp_rules.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
        with patch.object(case_rules, "_BRAND_PATTERNS_PATH", self._temp_rules):
            case_rules._BRAND_PATTERNS_CACHE = None
            result = case_rules._structured_fallback("VoltGate VG-11KW-Pro 故障码 C-RCD-04")
            self.assertEqual(result["brand"], "VoltGate")
            self.assertEqual(result["charger_model"], "VG-11KW-Pro")

    def test_extract_charger_model_falls_back_to_hardcoded(self) -> None:
        """JSON 缺失时型号提取使用 VG-* 硬编码兜底。"""
        with patch.object(case_rules, "_BRAND_PATTERNS_PATH", self._temp_rules):
            self._temp_rules.unlink(missing_ok=True)
            case_rules._BRAND_PATTERNS_CACHE = None
            self.assertEqual(case_rules._extract_charger_model("设备 VG-WallBox2 异常", []), "VG-WallBox2")
            self.assertEqual(case_rules._extract_charger_model("没有型号的文本", []), "")

    def test_structured_fallback_no_brand_in_text(self) -> None:
        """文本中无品牌名时 brand 为空字符串。"""
        with patch.object(case_rules, "_BRAND_PATTERNS_PATH", self._temp_rules):
            self._temp_rules.unlink(missing_ok=True)
            case_rules._BRAND_PATTERNS_CACHE = None
            result = case_rules._structured_fallback("充电桩坏了，屏幕不亮")
            self.assertEqual(result["brand"], "")
            self.assertEqual(result["charger_model"], "")


if __name__ == "__main__":
    unittest.main()
