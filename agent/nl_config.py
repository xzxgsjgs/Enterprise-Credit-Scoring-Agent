"""agent.nl_config — 自然语言 → config overrides 解析与校验（模块 4）。

设计要点：
- **LLM 严禁生成数值之外的东西**：只让它把中文需求映射成 {键: 值} JSON。
- **双保险**：LLM 输出后必须过 validate_patch（白名单 + 范围夹取 + 类型校验），
  越界项被拒绝并写入 warnings，绝不静默接受。
- **必须可降级**：无 .env / 无 key / 调用超时 / JSON 解析失败 → 返回 ({}, [warning])。
- 纯规则解析（_rule_parse）作为第二层兜底，覆盖高频表达，
  即使 LLM 完全不可用，常见中文需求仍能解析成功。

用法：
    patch, warnings = parse_nl_config("用2018年以后数据、最多分6箱、开启时序验证")
    # patch = {"min_year": 2018, "max_bins": 6, "time_split": True}
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# 合法键清单（与 config/default_config.yaml 的扁平化键对应）
# ============================================================================
# 结构：{键: (类型, 最小值, 最大值, 说明)}
NUMERIC_KEYS: dict[str, tuple[type, float, float, str]] = {
    "iv_threshold": (float, 0.0, 0.2, "IV 阈值，低于此值的变量被剔除"),
    "missing_threshold": (float, 0.1, 0.9, "缺失率上限，超出的变量被剔除"),
    "identical_threshold": (float, 0.5, 1.0, "单一值占比上限"),
    "max_bins": (int, 3, 12, "最大分箱数"),
    "min_bin_size": (float, 0.01, 0.2, "最小箱占比"),
    "outlier_sigma": (float, 2.0, 8.0, "异常值截尾的 sigma 倍数"),
    "regularization": (float, 0.001, 1.0, "正则化强度 1/C，越小正则化越强"),
    "test_size": (float, 0.1, 0.5, "测试集比例"),
    "min_ks": (float, 0.05, 0.6, "KS 护栏下限"),
    "min_auc": (float, 0.5, 0.95, "AUC 护栏下限"),
    "max_psi": (float, 0.05, 0.5, "PSI 护栏上限"),
    "max_iter": (int, 100, 10000, "逻辑回归最大迭代次数"),
    "min_year": (int, 1990, 2100, "只用 >= 该年份的数据"),
}

BOOL_KEYS: dict[str, str] = {
    "time_split": "是否启用时序切分（Expanding Window CV）",
    "use_lgbm": "是否训练 LGBM 对照模型",
    "require_human_review": "护栏警告时是否强制人工复核",
}

CHOICE_KEYS: dict[str, tuple[list[str], str]] = {
    "binning_method": (["tree", "chimerge", "quantile", "equal"], "分箱算法"),
    "missing_strategy": (["mean", "median", "mode", "constant", "drop"], "缺失值填充策略"),
    "outlier_method": (["cap", "clip", "remove", "none"], "异常值处理方法"),
}

LIST_KEYS: dict[str, str] = {
    "exclude_vars": "建模时排除的变量名列表",
    "exclude_industries": "排除的行业代码列表",
}

# 全部合法键
ALL_KEYS = set(NUMERIC_KEYS) | set(BOOL_KEYS) | set(CHOICE_KEYS) | set(LIST_KEYS)


# ============================================================================
# 校验层（纯确定性，可单测）
# ============================================================================
def validate_patch(patch: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """白名单 + 范围 + 类型校验，返回 (净化后的 patch, warnings)。

    规则：
      - 键不在 ALL_KEYS → 丢弃，记 warning
      - 数值键：类型转换 + 越界夹到边界，记 warning
      - 布尔键：接受 True/False/"true"/"false"/1/0
      - 选择键：不在 choices 内 → 丢弃，记 warning
      - 列表键：必须是 list 或可转成 list
    """
    clean: dict[str, Any] = {}
    warnings: list[str] = []

    if not isinstance(patch, dict):
        return {}, [f"patch 不是 dict: {type(patch).__name__}"]

    for key, raw_val in patch.items():
        # ---- 数值键 ----
        if key in NUMERIC_KEYS:
            typ, lo, hi, _desc = NUMERIC_KEYS[key]
            try:
                val = typ(raw_val) if typ is int else float(raw_val)
            except (TypeError, ValueError):
                warnings.append(f"[丢弃] {key}: 值 {raw_val!r} 无法转为 {typ.__name__}")
                continue
            if val < lo or val > hi:
                clamped = type(val)(max(lo, min(hi, val)))
                warnings.append(
                    f"[夹取] {key}: {val} 超出合法区间 [{lo}, {hi}]，已夹到 {clamped}"
                )
                val = clamped
            clean[key] = val
            continue

        # ---- 布尔键 ----
        if key in BOOL_KEYS:
            if isinstance(raw_val, bool):
                clean[key] = raw_val
            elif isinstance(raw_val, str):
                s = raw_val.strip().lower()
                if s in {"true", "1", "yes", "是", "开", "开启"}:
                    clean[key] = True
                elif s in {"false", "0", "no", "否", "关", "关闭"}:
                    clean[key] = False
                else:
                    warnings.append(f"[丢弃] {key}: 无法识别的布尔值 {raw_val!r}")
            elif isinstance(raw_val, (int, float)):
                clean[key] = bool(raw_val)
            else:
                warnings.append(f"[丢弃] {key}: 无法识别的布尔值 {raw_val!r}")
            continue

        # ---- 选择键 ----
        if key in CHOICE_KEYS:
            choices, _desc = CHOICE_KEYS[key]
            if raw_val in choices:
                clean[key] = raw_val
            else:
                warnings.append(
                    f"[丢弃] {key}: {raw_val!r} 不在合法选项 {choices} 内"
                )
            continue

        # ---- 列表键 ----
        if key in LIST_KEYS:
            if isinstance(raw_val, list):
                clean[key] = raw_val
            elif isinstance(raw_val, str):
                # 支持逗号 / 顿号分隔
                items = [x.strip() for x in re.split(r"[,，、;；]", raw_val) if x.strip()]
                clean[key] = items
            else:
                warnings.append(f"[丢弃] {key}: 需要列表，得到 {type(raw_val).__name__}")
            continue

        # ---- 未知键 ----
        warnings.append(f"[丢弃] 未知配置项: {key}")

    return clean, warnings


# ============================================================================
# 规则兜底解析（无 LLM 也能覆盖高频表达）
# ============================================================================
_RULE_PATTERNS: list[tuple[re.Pattern[str], str, Any]] = [
    # 年份
    (re.compile(r"(?:用|只用|取|筛选)\s*(\d{4})\s*年(?:份)?(?:以|之)?后"), "min_year", lambda m: int(m.group(1))),
    (re.compile(r"(\d{4})\s*年(?:份)?(?:以|之)?后"), "min_year", lambda m: int(m.group(1))),
    # 分箱
    (re.compile(r"(?:最多|至多|不超过)?\s*分\s*(\d+)\s*(?:箱|个箱|箱?)"), "max_bins", lambda m: int(m.group(1))),
    (re.compile(r"max_bins\s*[:=]\s*(\d+)"), "max_bins", lambda m: int(m.group(1))),
    # KS / AUC / PSI（动词可选，支持 "KS 0.4" / "KS 调到 0.99" / "ks: 0.35"）
    (re.compile(r"ks\s*(?:目标|要求|阈值|调到|设为|改为|调成|到|为)?\s*[:=]?\s*(0?\.\d+|1(?:\.0+)?)", re.I),
     "min_ks", lambda m: float(m.group(1))),
    (re.compile(r"auc\s*(?:目标|要求|阈值|调到|设为|改为|调成|到|为)?\s*[:=]?\s*(0?\.\d+|1(?:\.0+)?)", re.I),
     "min_auc", lambda m: float(m.group(1))),
    (re.compile(r"psi\s*(?:目标|要求|阈值|调到|设为|改为|调成|到|为)?\s*[:=]?\s*(0?\.\d+|1(?:\.0+)?)", re.I),
     "max_psi", lambda m: float(m.group(1))),
    # 正则化
    (re.compile(r"正则化\s*(?:强度)?\s*(?:调到|设为|改为|调成|到|为)?\s*(0?\.\d+)"),
     "regularization", lambda m: float(m.group(1))),
    # 测试集
    (re.compile(r"测试集(?:比例)?\s*(?:设为|改为|调到|到|为)?\s*(0?\.\d+)"),
     "test_size", lambda m: float(m.group(1))),
    # IV 阈值
    (re.compile(r"iv\s*(?:阈值)?\s*(?:设为|改为|调到|到|为)?\s*[:=]?\s*(0?\.\d+)", re.I),
     "iv_threshold", lambda m: float(m.group(1))),
    # 最小箱占比
    (re.compile(r"(?:最小)?箱占比\s*(?:设为|改为|调到|到|为)?\s*(0?\.\d+)"),
     "min_bin_size", lambda m: float(m.group(1))),
]

_RULE_BOOL: list[tuple[re.Pattern[str], str, bool]] = [
    (re.compile(r"(开启|启用|打开|使用)\s*时序(?:验证|切分|切)?"), "time_split", True),
    (re.compile(r"(关闭|禁用|不用)\s*时序(?:验证|切分|切)?"), "time_split", False),
    (re.compile(r"(开启|启用|加上|训练)\s*(?:lgbm|lightgbm|对照模型)", re.I), "use_lgbm", True),
    (re.compile(r"(关闭|不要|取消)\s*(?:lgbm|lightgbm|对照模型)", re.I), "use_lgbm", False),
]

_INDUSTRY_MAP: dict[str, str] = {
    "房地产": "K70",
    "地产": "K70",
    "银行": "J66",
    "证券": "J67",
    "保险": "J68",
    "医药": "C27",
    "汽车": "C36",
}


def _rule_parse(text: str) -> dict[str, Any]:
    """纯规则解析中文需求 → patch（不含任何 LLM 调用）。"""
    patch: dict[str, Any] = {}

    for pat, key, conv in _RULE_PATTERNS:
        m = pat.search(text)
        if m and key not in patch:
            try:
                patch[key] = conv(m)
            except Exception:  # noqa: BLE001 - 正则转换失败就跳过
                pass

    for pat, key, val in _RULE_BOOL:
        if pat.search(text) and key not in patch:
            patch[key] = val

    # 排除行业
    for zh, code in _INDUSTRY_MAP.items():
        if re.search(rf"(排除|剔除|去掉|不要)\s*{zh}", text):
            patch.setdefault("exclude_industries", [])
            if code not in patch["exclude_industries"]:
                patch["exclude_industries"].append(code)

    return patch


# ============================================================================
# 主入口
# ============================================================================
def _build_prompt(text: str) -> str:
    lines = [
        "你是信用风险建模的配置助手。把下面的中文需求映射为 JSON 配置补丁。",
        "",
        "【合法键清单】只允许使用以下键，不得自创：",
    ]
    for k, (_t, lo, hi, desc) in NUMERIC_KEYS.items():
        lines.append(f"  - {k} (数值, 范围 [{lo}, {hi}]): {desc}")
    for k, desc in BOOL_KEYS.items():
        lines.append(f"  - {k} (布尔 true/false): {desc}")
    for k, (choices, desc) in CHOICE_KEYS.items():
        lines.append(f"  - {k} (枚举 {choices}): {desc}")
    for k, desc in LIST_KEYS.items():
        lines.append(f"  - {k} (数组): {desc}")
    lines += [
        "",
        "【输出要求】",
        "1. 只输出一个 JSON 对象，不要任何解释文字，不要 markdown 代码块。",
        "2. 无法映射的需求直接忽略，不要输出对应键。",
        "3. 例：输入『用2018年以后数据、最多分6箱、开启时序验证』",
        "   输出：{\"min_year\": 2018, \"max_bins\": 6, \"time_split\": true}",
        "",
        f"【待解析需求】{text}",
    ]
    return "\n".join(lines)


def parse_nl_config(text: str, use_llm: bool = True) -> tuple[dict[str, Any], list[str]]:
    """把中文需求解析为 config_overrides。

    Args:
        text: 中文自然语言需求
        use_llm: False 时只用规则解析（测试/降级用）

    Returns:
        (patch, warnings)
        - patch 已通过 validate_patch 白名单与范围校验
        - warnings 记录被丢弃/夹取的项，以及 LLM 不可用的提示
    """
    if not text or not text.strip():
        return {}, []

    warnings: list[str] = []
    raw: dict[str, Any] | None = None

    # ---- 1) LLM 解析（可失败）----
    if use_llm:
        raw = _llm_json(_build_prompt(text))
        if raw is None:
            warnings.append("LLM 不可用或返回非法 JSON，已回退到规则解析")
        elif not isinstance(raw, dict):
            warnings.append(f"LLM 返回非 dict（{type(raw).__name__}），已回退到规则解析")
            raw = None

    # ---- 2) 规则兜底 / 补充 ----
    rule_patch = _rule_parse(text)
    if raw is None:
        merged = rule_patch
    else:
        # LLM 结果为主，规则结果补充 LLM 漏掉的键
        merged = {**rule_patch, **raw}

    if not merged:
        warnings.append("未能从需求中解析出任何合法配置项")
        return {}, warnings

    # ---- 3) 校验 ----
    clean, vw = validate_patch(merged)
    warnings.extend(vw)
    logger.info("NL 配置解析: %r → %s (warnings=%d)", text, clean, len(warnings))
    return clean, warnings


def _llm_json(prompt: str) -> dict[str, Any] | None:
    """调 LLM 并解析 JSON；任何异常返回 None。"""
    try:
        from app.llm.client import get_model

        model = get_model()
        resp = model.invoke(prompt)
        content = getattr(resp, "content", "") or ""
        cleaned = content.strip()
        # 容忍 ```json ... ``` 包裹
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        # 提取第一个 {...}
        if not cleaned.startswith("{"):
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                cleaned = m.group(0)
        return json.loads(cleaned)
    except Exception as e:  # noqa: BLE001
        logger.warning("NL 配置 LLM 解析失败: %s", type(e).__name__)
        return None