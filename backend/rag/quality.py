from __future__ import annotations

import re

"""文档解析后的质量检测器"""

def analyze_digit_health(pages: list[dict]) -> dict:
    text = "\n".join(page.get("text", "") for page in pages)
    total_chars = len(text) # 计算总字符数
    digit_count = len(re.findall(r"\d", text)) # 计算数字字符数量
    year_count = len(re.findall(r"(?:19|20)\d{2}", text)) # 计算年份数量
    percent_count = len(re.findall(r"\d+(?:\.\d+)?\s*%", text)) # 计算百分比数量
    error_code_count = len(re.findall(r"\b[A-Z]\d{2,4}\b", text, re.I)) # 计算错误码数量
    model_count = len(re.findall(r"\b[A-Z]{2,4}[-_ ]?\d{2,5}\b", text, re.I)) # 计算型号数量
    money_count = len(re.findall(r"(?:￥|¥|元|RMB|\$)\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*元", text, re.I)) # 计算金额数量
    warranty_period_count = len(
        re.findall(r"\d+\s*(?:个月|月|年)|半年|一年|一年半|两年|二年", text)
    ) # 计算保修期表达数量

    digit_ratio = digit_count / total_chars if total_chars else 0 # 计算数字占比
    warnings = []

    if total_chars >= 500 and digit_count < 10:
        warnings.append("解析结果中的数字数量异常偏少，可能存在数字丢失。")
    if total_chars >= 500 and digit_ratio < 0.005:
        warnings.append("数字占比过低，建议对照原 PDF 检查年份、页码、表格编号和百分比。")
    if total_chars >= 300 and not (error_code_count or model_count or warranty_period_count):
        warnings.append("未检测到错误码、型号/SKU 或保障期表达；若文档为售后资料，请检查解析完整性。")

    return {
        "total_chars": total_chars,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "year_count": year_count,
        "percent_count": percent_count,
        "error_code_count": error_code_count,
        "model_count": model_count,
        "money_count": money_count,
        "warranty_period_count": warranty_period_count,
        "warnings": warnings,
    }
