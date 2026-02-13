from __future__ import annotations

import json
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------
# Config: stages & parsing rules
# -----------------------------
TRIGGER_STAGES = {"知识协作类", "知识整合类", "知识规划类"}
OUTCOME_STAGE = "知识重构类"
EPS = 1e-6  # smoothing for log-lift and divide-by-zero safety

# split delimiters for multi-actor strings
_SPLIT_RE = re.compile(r"[、,，;；/／\s]+|和|及|与|以及|或")


# -----------------------------
# Parsing a.txt / c.txt with duplicate keys preserved
# (We align a.txt and c.txt to v.txt by ORDER of appearance in the raw text.)
# -----------------------------
def _extract_data_blocks(raw_text: str) -> Dict[str, str]:
    """Extract per-data blocks like \"dataX\": { ... } from raw text."""
    blocks: Dict[str, str] = {}
    for m in re.finditer(r'"(data\d+)"\s*:\s*\{', raw_text):
        key = m.group(1)
        start = m.end() - 1  # at '{'
        depth = 0
        i = start
        while i < len(raw_text):
            ch = raw_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blocks[key] = raw_text[start:i + 1]
                    break
            i += 1
    return blocks


def _parse_c_block(block: str) -> List[Tuple[str, str]]:
    """Parse c.txt block as ordered list of (verb, stage) pairs."""
    return re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block)


def _parse_a_block(block: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse a.txt block as ordered list of (verb, attr_dict) pairs."""
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    i = 0
    n = len(block)
    while i < n:
        m = re.search(r'"([^"]+)"\s*:\s*\{', block[i:])
        if not m:
            break
        verb = m.group(1)
        obj_start = i + m.end() - 1  # '{'
        depth = 0
        j = obj_start
        while j < n:
            if block[j] == "{":
                depth += 1
            elif block[j] == "}":
                depth -= 1
                if depth == 0:
                    obj_text = block[obj_start:j + 1]
                    try:
                        attrs = json.loads(obj_text)
                    except json.JSONDecodeError:
                        obj_text2 = re.sub(r",\s*}", "}", obj_text)
                        obj_text2 = re.sub(r",\s*]", "]", obj_text2)
                        attrs = json.loads(obj_text2)
                    pairs.append((verb, attrs))
                    i = j + 1
                    break
            j += 1
        else:
            break
    return pairs


# -----------------------------
# Mapping & normalization
# -----------------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in df.columns:
        if any(k in str(c) for k in candidates):
            return c
    raise ValueError(f"Cannot find column among candidates {candidates}. Found columns: {list(df.columns)}")


def _norm_executor(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    s = s.replace("\u3000", " ").strip()
    if not s:
        return None
    s = re.sub(r"(等人|等)$", "", s).strip()
    return s if s else None


def _split_actors(executor: str) -> List[str]:
    executor = executor.strip()
    parts = [p.strip() for p in _SPLIT_RE.split(executor) if p and p.strip()]
    return parts if parts else [executor]


@dataclass
class RoleMap:
    exec2role: Dict[str, int]
    strategic_names: set

    @classmethod
    def from_excel(cls, path: Path) -> "RoleMap":
        """
        Robustly load executor->role mapping.

        Why: some workbooks include columns like "阶段类别" whose cells are strings
        (e.g., "知识规划类"). A naive "contains('类别')" match may pick the wrong column.

        Strategy:
        1) Pick executor name column: prefer exact "执行者", else any column containing "执行者"
           but NOT containing "类别".
        2) Pick role column:
           - Prefer columns named like "执行者类别"/"角色"/"角色标注".
           - Otherwise, among candidate columns whose name contains "类别" (but not "阶段"),
             choose the one that can be coerced to numeric and whose values mostly fall in {0,1,2,3}.
        """
        df = pd.read_excel(path)

        # ---- executor column ----
        col_exec = None
        if "执行者" in df.columns:
            col_exec = "执行者"
        else:
            exec_cands = [c for c in df.columns if ("执行者" in str(c) and "类别" not in str(c))]
            if exec_cands:
                col_exec = exec_cands[0]
        if col_exec is None:
            # fallback: any column containing 执行者
            exec_cands = [c for c in df.columns if "执行者" in str(c)]
            if exec_cands:
                col_exec = exec_cands[0]
        if col_exec is None:
            raise ValueError(f"Cannot find executor column. Columns={list(df.columns)}")

        # ---- role column ----
        preferred_role_names = [
            "执行者类别", "执行者类别(0/1/2/3)", "角色", "角色标注", "执行者角色", "角色类别", "类别(执行者)"
        ]
        col_role = None
        for name in preferred_role_names:
            if name in df.columns:
                col_role = name
                break

        def _score_role_col(series: pd.Series) -> tuple:
            """Higher is better: (in_set_ratio, non_na_ratio, unique_count)"""
            s_num = pd.to_numeric(series, errors="coerce")
            non_na = s_num.notna().mean()
            if non_na == 0:
                return (0.0, 0.0, 999)
            in_set = s_num.dropna().isin([0, 1, 2, 3]).mean()
            uniq = s_num.dropna().nunique()
            return (float(in_set), float(non_na), int(uniq))

        if col_role is None:
            # candidates containing 类别 but not 阶段 (avoid picking stage category)
            role_cands = [
                c for c in df.columns
                if ("类别" in str(c)) and ("阶段" not in str(c)) and ("说明" not in str(c))
            ]
            # also consider any other columns that might contain role codes
            if not role_cands:
                role_cands = [c for c in df.columns if ("角色" in str(c)) or ("类别" in str(c))]

            best = None
            best_score = (-1.0, -1.0, 999)
            for c in role_cands:
                score = _score_role_col(df[c])
                # we want mostly {0,1,2,3}, decent coverage, and small unique count
                if score > best_score:
                    best_score = score
                    best = c
            col_role = best

        if col_role is None:
            raise ValueError(f"Cannot find role/category column. Columns={list(df.columns)}")

        # ---- build mapping ----
        out = df[[col_exec, col_role]].copy()
        out[col_exec] = out[col_exec].astype(str).str.strip()
        out[col_role] = pd.to_numeric(out[col_role], errors="coerce")
        out = out.dropna(subset=[col_exec, col_role])

        # Keep only 0/1/2/3 if present (safety)
        out = out[out[col_role].isin([0, 1, 2, 3])]
        out[col_role] = out[col_role].astype(int)

        exec2role = dict(zip(out[col_exec], out[col_role]))
        strategic = {k for k, v in exec2role.items() if v == 3}

        print(f"[INFO] Mapping loaded from {path.name}: executor_col={col_exec}, role_col={col_role}, n={len(exec2role)}")
        return cls(exec2role=exec2role, strategic_names=strategic)


    def role_of(self, executor: Optional[str]) -> Optional[int]:
        if executor is None:
            return None
        if executor in self.exec2role:
            return int(self.exec2role[executor])
        parts = _split_actors(executor)
        roles = [self.exec2role.get(p) for p in parts if p in self.exec2role]
        if roles:
            return int(max(roles))
        return None

    def strategic_in(self, executor: Optional[str]) -> List[str]:
        if executor is None:
            return []
        parts = _split_actors(executor)
        return [p for p in parts if p in self.strategic_names]


# -----------------------------
# Core computation
# -----------------------------
def load_inputs(input_dir: Path):
    v_path = input_dir / "v.txt"
    a_path = input_dir / "a.txt"
    c_path = input_dir / "c.txt"

    v_data = json.loads(v_path.read_text(encoding="utf-8").strip())

    a_raw = a_path.read_text(encoding="utf-8")
    c_raw = c_path.read_text(encoding="utf-8")

    a_blocks = _extract_data_blocks(a_raw)
    c_blocks = _extract_data_blocks(c_raw)

    c_pairs = {k: _parse_c_block(c_blocks[k]) for k in v_data.keys()}
    a_pairs = {k: _parse_a_block(a_blocks[k]) for k in v_data.keys()}

    mism = []
    for d in v_data.keys():
        lv = len(v_data[d])
        lc = len(c_pairs[d])
        la = len(a_pairs[d])
        if not (lv == lc == la):
            mism.append((d, lv, lc, la))
    if mism:
        print("[WARN] Length mismatch found (data, len_v, len_c, len_a):")
        for row in mism[:20]:
            print("   ", row)
        print("These cases may break alignment between v/a/c. Please inspect inputs.")
    return v_data, c_pairs, a_pairs


def build_event_sequences(v_data, c_pairs, a_pairs, role_map: RoleMap) -> pd.DataFrame:
    rows = []
    missing_executor = 0
    missing_role_counter = Counter()

    for data, verbs in v_data.items():
        seq_idx = 0
        for i, verb in enumerate(verbs):
            stage = c_pairs[data][i][1]
            attrs = a_pairs[data][i][1]
            executor = _norm_executor(attrs.get("执行者"))
            if executor is None:
                missing_executor += 1
                continue
            role = role_map.role_of(executor)
            if role is None:
                missing_role_counter[executor] += 1
            rows.append({
                "data": data,
                "t": seq_idx,
                "t_raw": i,
                "verb": verb,
                "stage": stage,
                "executor": executor,
                "role": role,
                "strategic_names": role_map.strategic_in(executor),
            })
            seq_idx += 1

    events = pd.DataFrame(rows).sort_values(["data", "t"]).reset_index(drop=True)

    print(f"[INFO] Total original verb nodes: {sum(len(v) for v in v_data.values())}")
    print(f"[INFO] Dropped nodes without explicit executor: {missing_executor}")
    print(f"[INFO] Remaining events (with executor): {len(events)}")

    if missing_role_counter:
        top = missing_role_counter.most_common(15)
        print("[WARN] Executors not found in role mapping (top 15 among kept events):")
        for name, cnt in top:
            print(f"   {name}: {cnt}")
        print("You may want to update 执行者统计_角色标注.xlsx or adjust normalization rules.")
    return events


def compute_outcome(events_case: pd.DataFrame, delta: int) -> np.ndarray:
    stage = events_case["stage"].to_numpy()
    role = events_case["role"].to_numpy()

    is_recon = (stage == OUTCOME_STAGE) & (pd.notna(role)) & (role > 1)
    arr = is_recon.astype(np.int8)
    n = len(arr)

    pref = np.zeros(n + 1, dtype=np.int32)
    pref[1:] = np.cumsum(arr)

    idx = np.arange(n, dtype=np.int32)
    j1 = idx + 1
    j2 = np.minimum(n, idx + delta + 1)
    win = (pref[j2] - pref[j1]) > 0
    return win.astype(np.int8)


def lifts_by_delta(events: pd.DataFrame, max_delta: int, role_map: RoleMap, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    is_trigger_stage = events["stage"].isin(TRIGGER_STAGES)
    is_T = (events["role"] == 3) & is_trigger_stage
    is_B = (pd.notna(events["role"])) & (events["role"] != 3) & is_trigger_stage

    # scientist context: cases where each strategic scientist appears
    scientist_cases = defaultdict(set)
    for data, g in events[events["role"] == 3].groupby("data"):
        for names in g["strategic_names"]:
            for s in names:
                scientist_cases[s].add(data)

    trig_df = events[is_T].copy().explode("strategic_names")
    trig_df = trig_df.rename(columns={"strategic_names": "scientist"})
    trig_df = trig_df[pd.notna(trig_df["scientist"])]

    summary_rows = []
    case_rows_all = []
    sci_rows_all = []

    for delta in range(1, max_delta + 1):
        outcomes = np.zeros(len(events), dtype=np.int8)
        for _, idxs in events.groupby("data").groups.items():
            g = events.loc[idxs].sort_values("t")
            out = compute_outcome(g, delta)
            outcomes[g.index.to_numpy()] = out

        events_delta = events.copy()
        events_delta["Y"] = outcomes

        # pooled
        pT = events_delta.loc[is_T, "Y"].mean() if is_T.any() else np.nan
        pB = events_delta.loc[is_B, "Y"].mean() if is_B.any() else np.nan
        pooled_lift = (pT / pB) if (pd.notna(pT) and pd.notna(pB) and pB > 0) else np.nan

        # per-case
        per_case = []
        for data, g in events_delta.groupby("data"):
            Tg = g[is_T.loc[g.index]]
            Bg = g[is_B.loc[g.index]]
            if len(Tg) == 0 or len(Bg) == 0:
                continue
            pT_d = float(Tg["Y"].mean())
            pB_d = float(Bg["Y"].mean())
            lift_d = (pT_d / pB_d) if pB_d > 0 else np.nan
            per_case.append({
                "delta": delta,
                "data": data,
                "n_T": int(len(Tg)),
                "n_B": int(len(Bg)),
                "pT": pT_d,
                "pB": pB_d,
                "lift": lift_d,
                "log_lift_eps": math.log((pT_d + EPS) / (pB_d + EPS)),
            })
        case_df = pd.DataFrame(per_case)
        case_rows_all.append(case_df)

        if len(case_df):
            case_ratio_of_means = (case_df["pT"].mean() / case_df["pB"].mean()) if case_df["pB"].mean() > 0 else np.nan
            case_geo_lift = float(math.exp(case_df["log_lift_eps"].mean()))
        else:
            case_ratio_of_means = np.nan
            case_geo_lift = np.nan

        # per-scientist (context-matched baseline)
        per_sci = []
        for sci, gT in trig_df.groupby("scientist"):
            cases = scientist_cases.get(sci, set())
            if not cases:
                continue
            idx_T = gT.index
            pT_s = float(events_delta.loc[idx_T, "Y"].mean()) if len(idx_T) else np.nan

            idx_B = events_delta.index[is_B & events_delta["data"].isin(cases)]
            if len(idx_B) == 0:
                continue
            pB_s = float(events_delta.loc[idx_B, "Y"].mean())
            lift_s = (pT_s / pB_s) if pB_s > 0 else np.nan

            per_sci.append({
                "delta": delta,
                "scientist": sci,
                "n_T": int(len(idx_T)),
                "n_B": int(len(idx_B)),
                "pT": pT_s,
                "pB": pB_s,
                "lift": lift_s,
                "log_lift_eps": math.log((pT_s + EPS) / (pB_s + EPS)),
                "n_cases_context": int(len(cases)),
            })
        sci_df = pd.DataFrame(per_sci)
        sci_rows_all.append(sci_df)

        if len(sci_df):
            sci_ratio_of_means = (sci_df["pT"].mean() / sci_df["pB"].mean()) if sci_df["pB"].mean() > 0 else np.nan
            sci_geo_lift = float(math.exp(sci_df["log_lift_eps"].mean()))
        else:
            sci_ratio_of_means = np.nan
            sci_geo_lift = np.nan

        summary_rows.append({
            "delta": delta,
            "pooled_pT": pT,
            "pooled_pB": pB,
            "pooled_lift": pooled_lift,
            "case_geo_lift_eps": case_geo_lift,
            "case_ratio_of_means": case_ratio_of_means,
            "n_cases_used": int(case_df["data"].nunique()) if len(case_df) else 0,
            "sci_geo_lift_eps": sci_geo_lift,
            "sci_ratio_of_means": sci_ratio_of_means,
            "n_scientists_used": int(sci_df["scientist"].nunique()) if len(sci_df) else 0,
            "n_T_events_total": int(is_T.sum()),
            "n_B_events_total": int(is_B.sum()),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "lift_by_delta.csv", index=False, encoding="utf-8-sig")

    if case_rows_all:
        pd.concat(case_rows_all, ignore_index=True).to_csv(output_dir / "case_lift_by_delta.csv", index=False, encoding="utf-8-sig")
    if sci_rows_all:
        pd.concat(sci_rows_all, ignore_index=True).to_csv(output_dir / "scientist_lift_by_delta.csv", index=False, encoding="utf-8-sig")

    return summary


def plot_lift_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    x = summary["delta"].to_numpy()[:8]

    plt.figure()
    # plt.plot(x, summary["pooled_lift"][:8], marker="o", label="Pooled Lift")
    plt.plot(x, summary["case_geo_lift_eps"][:8], marker="o", label="案例等权")
    # plt.plot(x, summary["sci_geo_lift_eps"][:8], marker="o")
    plt.axhline(1.0, linewidth=1)
    plt.xlabel("Δ (steps)")
    plt.ylabel("Lift(Δ)")
    plt.title("战略科学家倍增指数Lift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lift_curves.png", dpi=200)
    plt.show()


def main():
    # ==================== 配置区域：在此修改文件路径和参数 ====================
    # v.txt / a.txt / c.txt 所在的目录（这三个文件需要在同一目录下）
    INPUT_DIR = Path(r"data\result\gold")
    
    # 执行者角色映射表的路径（Excel 文件）
    MAPPING_PATH = Path(r"data\excel\执行者统计_角色标注.xlsx")
    
    # 输出目录
    OUTPUT_DIR = Path(r"data\excel")
    
    # 计算 Lift 的最大时间窗口 Δ（计算 Δ=1..max_delta）
    MAX_DELTA = 12
    # ========================================================================

    # 验证路径是否存在
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"映射文件不存在: {MAPPING_PATH}")
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    role_map = RoleMap.from_excel(MAPPING_PATH)
    v_data, c_pairs, a_pairs = load_inputs(INPUT_DIR)

    events = build_event_sequences(v_data, c_pairs, a_pairs, role_map)
    summary = lifts_by_delta(events, MAX_DELTA, role_map, OUTPUT_DIR)

    print("\n=== Lift(Δ) summary (first rows) ===")
    print(summary.head(MAX_DELTA).to_string(index=False))

    plot_lift_curves(summary, OUTPUT_DIR)

    print(f"\n[DONE] Outputs saved to: {OUTPUT_DIR}")
    print(" - lift_by_delta.csv")
    print(" - case_lift_by_delta.csv")
    print(" - scientist_lift_by_delta.csv")
    print(" - lift_curves.png")


if __name__ == "__main__":
    main()
