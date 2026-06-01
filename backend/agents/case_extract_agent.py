from __future__ import annotations

import re
from typing import Any

from backend.schemas import AfterSalesCase


class CaseExtractAgent:
    """从客户售后文本中提取结构化字段."""

    def extract(self, text: str) -> dict[str, Any]:
        text = text or ""
        case = AfterSalesCase(
            product_model=self._first_match(text, r"\b([A-Z]{1,4}[-_ ]?\d{2,5})\b").replace(" ", "-"),  # 产品型号
            fault_code=self._first_match(text, r"\b(E\d{2,4}|F\d{2,4}|P\d{2,4})\b").upper(),  # 故障码 
            symptoms=self._extract_symptoms(text), # 故障现象
            purchase_time=self._extract_purchase_time(text), # 购买时间
            city=self._extract_city(text), # 城市
            phone=self._first_match(text, r"1[3-9]\d{9}"), # 联系方式
            address=self._extract_address(text), # 地址
            has_water_leak=any(word in text for word in ["漏水", "渗水", "滴水", "插座打湿"]), # 是否漏水
            has_power_issue=any(word in text for word in ["漏电", "插座打湿", "冒烟", "烧焦", "跳闸", "无法开机"]), # 是否有电力问题
            has_restarted=any(word in text for word in ["重启", "断电重启", "重新开机"]), # 是否重启过
            complaint_intent=any(word in text for word in ["投诉", "差评", "主管"]), # 是否有投诉意向
            refund_intent=any(word in text for word in ["退款", "退货", "赔偿"]), # 是否有退款意向
            raw_text=text,  #原文本
        )
        case.missing_info = self._missing_info(case)
        return case.to_dict()

    def _first_match(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.I)
        return match.group(1).strip() if match else ""

    def _extract_symptoms(self, text: str) -> list[str]:
        keywords = [
            "出水慢",
            "出水变慢",
            "不出水",
            "漏水",
            "渗水",
            "插座打湿",
            "漏电",
            "冒烟",
            "烧焦味",
            "跳闸",
            "显示故障",
            "加热异常",
            "水质异常",
            "噪音大",
            "无法开机",
            "无法营业",
        ]
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            return hits

        if len(text) <= 80:
            return [text.strip()] if text.strip() else []
        return []

    def _extract_city(self, text: str) -> str:
        match = re.search(r"(北京|上海|广州|深圳|惠州|东莞|佛山|中山|珠海|镇江|苏州|杭州|成都|武汉|南京)", text)
        return match.group(1) if match else ""

    def _extract_purchase_time(self, text: str) -> str:
        patterns = [
            r"买了?[^，。；\s]{0,8}",
            r"购买[^，。；\s]{0,10}",
            r"\d+\s*(年|个月|月)",
            r"半年",
            r"一年",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""

    def _extract_address(self, text: str) -> str:
        match = re.search(r"(地址|门店|店铺)[:：]?\s*([^，。；]+)", text)
        return match.group(2).strip() if match else ""

    def _missing_info(self, case: AfterSalesCase) -> list[str]:
        missing = []
        if not case.product_model:
            missing.append("产品型号")
        if not case.symptoms:
            missing.append("故障现象")
        if not case.purchase_time:
            missing.append("购买时间")
        if not case.phone:
            missing.append("联系方式")
        if not case.address:
            missing.append("地址")
        return missing
