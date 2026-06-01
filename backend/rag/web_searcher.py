from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


def web_search(query, max_results=3):
    if not query or not query.strip():
        return []

    results = _search_bing_rss(query, max_results=max_results)

    if results:
        return results

    return [{
        "title": "联网搜索失败",
        "url": "",
        "snippet": "未能从 Bing RSS 获取搜索结果。请检查网络连接，或稍后重试。",
        "source": "web_error",
    }]


def _search_bing_rss(query, max_results=3):
    url = (
        "https://www.bing.com/search"
        f"?q={quote_plus(query)}"
        "&format=rss"
        "&mkt=zh-CN"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        results = []

        for item in root.findall(".//item"):
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            description = item.findtext("description", default="").strip()

            if not title and not description:
                continue

            results.append({
                "title": title,
                "url": link,
                "snippet": description,
                "source": "web",
            })

            if len(results) >= max_results:
                break

        return results

    except Exception:
        return []


def build_web_context(search_results):
    if not search_results:
        return ""

    parts = []

    for index, item in enumerate(search_results, start=1):
        if item.get("source") == "web_error":
            continue

        parts.append(
            f"联网资料 {index}\n"
            f"标题：{item.get('title', '')}\n"
            f"链接：{item.get('url', '')}\n"
            f"摘要：{item.get('snippet', '')}"
        )

    return "\n\n---\n\n".join(parts)


def merge_rag_and_web_context(rag_context, web_context):
    if not web_context:
        return rag_context

    return f"""
本地知识库资料：
{rag_context}

---

联网搜索补充资料：
{web_context}
""".strip()