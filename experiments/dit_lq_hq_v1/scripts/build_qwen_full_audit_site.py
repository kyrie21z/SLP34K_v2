#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


PROJECT_ROOT = Path("/mnt/data/zyx/SLP34K_v2").resolve()
EXPERIMENT_ROOT = (PROJECT_ROOT / "experiments" / "dit_lq_hq_v1").resolve()
SITE_DIR = (EXPERIMENT_ROOT / "qwen_full_audit_site").resolve()
APP_PATH = SITE_DIR / "app.py"
REPORT_PATH = SITE_DIR / "reports" / "Stage00_step12_qwen_full_audit_site_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="初始化 Qwen full audit site")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--qwen-parsed-csv", type=Path, default=EXPERIMENT_ROOT / "qwen_vl_hq_review_full/qwen_full_hq_selection_parsed.csv")
    parser.add_argument("--need-human-csv", type=Path, default=EXPERIMENT_ROOT / "qwen_vl_hq_review_full/qwen_full_need_human_review.csv")
    parser.add_argument("--candidate-groups", type=Path, default=EXPERIMENT_ROOT / "manual_hq_review/candidate_groups.csv")
    parser.add_argument("--candidate-samples", type=Path, default=EXPERIMENT_ROOT / "manual_hq_review/candidate_samples.csv")
    parser.add_argument("--review-images-root", type=Path, default=EXPERIMENT_ROOT / "manual_hq_review/images")
    parser.add_argument("--qwen-panels-dir", type=Path, default=EXPERIMENT_ROOT / "qwen_vl_hq_review_full/panels")
    parser.add_argument("--manual-selection-primary", type=Path, default=EXPERIMENT_ROOT / "manual_hq_review_site/data/manual_hq_selection_export.csv")
    parser.add_argument("--manual-selection-fallback", type=Path, default=EXPERIMENT_ROOT / "manual_hq_review_site/data/manual_hq_selection_live.csv")
    parser.add_argument("--db-path", type=Path, default=SITE_DIR / "data/qwen_audit_state.sqlite")
    parser.add_argument("--live-csv", type=Path, default=SITE_DIR / "data/qwen_full_audit_live.csv")
    parser.add_argument("--export-csv", type=Path, default=SITE_DIR / "data/qwen_full_audit_export.csv")
    parser.add_argument("--template", type=Path, default=SITE_DIR / "templates/index.html")
    parser.add_argument("--static-root", type=Path, default=SITE_DIR / "static")
    parser.add_argument("--no-auto-accept-p3", action="store_true")
    return parser.parse_args()


def load_app_module():
    spec = importlib.util.spec_from_file_location("qwen_full_audit_site_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 app.py: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        host=args.host,
        port=args.port,
        qwen_parsed_csv=args.qwen_parsed_csv.resolve(),
        need_human_csv=args.need_human_csv.resolve(),
        candidate_groups=args.candidate_groups.resolve(),
        candidate_samples=args.candidate_samples.resolve(),
        review_images_root=args.review_images_root.resolve(),
        qwen_panels_dir=args.qwen_panels_dir.resolve(),
        manual_selection_primary=args.manual_selection_primary.resolve(),
        manual_selection_fallback=args.manual_selection_fallback.resolve(),
        db_path=args.db_path.resolve(),
        live_csv=args.live_csv.resolve(),
        export_csv=args.export_csv.resolve(),
        template=args.template.resolve(),
        static_root=args.static_root.resolve(),
        no_auto_accept_p3=args.no_auto_accept_p3,
        smoke_test=False,
        init_only=False,
    )


def write_report(report_path: Path, args: argparse.Namespace, progress: Dict[str, Any], smoke_result: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    priority = progress["priority_counts"]
    status = progress["status_counts"]
    report = f"""# Stage00_step12 Qwen Full Audit Site Report

## 1. 范围与隔离声明

本轮只在 experiments/dit_lq_hq_v1/qwen_full_audit_site/ 下创建复核网站、SQLite 状态库、导出 CSV、README 和报告。
未修改原人工审核网站状态、Qwen 全量结果、Qwen pilot/mismatch 结果、v1/v2 manifest、原始数据、OCR LMDB、configs、checkpoints、outputs 或源码。

## 2. 输入与输出

- qwen parsed CSV: `{args.qwen_parsed_csv.resolve()}`
- qwen need-human CSV: `{args.need_human_csv.resolve()}`
- candidate_groups.csv: `{args.candidate_groups.resolve()}`
- candidate_samples.csv: `{args.candidate_samples.resolve()}`
- review images root: `{args.review_images_root.resolve()}`
- qwen panels dir: `{args.qwen_panels_dir.resolve()}`
- site dir: `{SITE_DIR}`
- app: `{APP_PATH}`
- sqlite: `{args.db_path.resolve()}`
- live CSV: `{args.live_csv.resolve()}`
- export CSV: `{args.export_csv.resolve()}`

## 3. Audit Priority 规则

- `P0`: need-human CSV、`json_parse_ok=False`、`illegal_selection=True`、selected sample `ocr_correct=False`、`ambiguous/no_clear_hq/layout_conflict`
- `P1`: 有 easy 却选 middle/hard；majority vertical 却选非 vertical；结构多样且未选 majority structure；`group_size > 16` 且既不是 v1 也不是 v2
- `P2`: 既不是 v1 也不是 v2；`confidence < 0.90`；`group_size > 20`；`group_size > 16`
- `P3`: 不属于 P0/P1/P2

## 4. 初始化统计

- `num_total_groups`: `{progress['total_groups']}`
- `P0 count`: `{priority['P0']}`
- `P1 count`: `{priority['P1']}`
- `P2 count`: `{priority['P2']}`
- `P3 count`: `{priority['P3']}`
- `pending initialized count`: `{status['pending']}`
- `reviewed initialized count`: `{status['reviewed']}`
- `skipped initialized count`: `{status['skipped']}`
- `auto_accept initialized count`: `{status['auto_accept']}`
- `no_auto_accept_p3`: `{args.no_auto_accept_p3}`

## 5. SQLite 状态表

数据库：

`qwen_full_audit_site/data/qwen_audit_state.sqlite`

状态字段包括：

- `label_hash`
- `label`
- `group_rank`
- `group_size`
- `qwen_hq_index`
- `final_hq_index`
- `review_status`
- `audit_priority`
- `audit_reasons`
- `review_decision`
- `review_note`
- `updated_at`

初始化策略：

- `P0/P1/P2 -> pending`
- `P3 -> auto_accept`，`final_hq_index = qwen_hq_index`
- 如果数据库已存在，则保留已有人工状态，只同步元数据和新增 group

## 6. API 列表

- `GET /`
- `GET /api/groups?priority=...&status=...&page=...&page_size=...`
- `GET /api/group/{{label_hash}}`
- `GET /api/progress`
- `POST /api/accept_qwen`
- `POST /api/select_hq`
- `POST /api/skip_group`
- `POST /api/export`

## 7. 前端交互说明

- 左侧 group 列表显示：`rank`、`label`、`priority`、`status`、`audit reasons`
- 右侧详情显示：Qwen panel、group 元信息、候选图 grid
- 支持：
  - `Accept Qwen`
  - 改选其他候选为 `final_hq_index`
  - `Skip`
  - `Previous / Next`
  - `P0/P1/P2/P3` 筛选
  - `pending/reviewed/skipped/auto_accept` 筛选

## 8. CSV 导出说明

Live CSV：

`qwen_full_audit_site/data/qwen_full_audit_live.csv`

Export CSV：

`qwen_full_audit_site/data/qwen_full_audit_export.csv`

字段：

- `label,label_hash,group_rank,group_size`
- `qwen_hq_index,final_hq_index`
- `review_status,audit_priority,audit_reasons`
- `review_decision,review_note,updated_at`
- `qwen_confidence,qwen_risk_flags`
- `qwen_quality,qwen_structure`
- `v1_hq_index,v2_hq_index`

## 9. 启动方式

```bash
cd /mnt/data/zyx/SLP34K_v2
conda activate slpr_ocr
python experiments/dit_lq_hq_v1/qwen_full_audit_site/app.py --host 127.0.0.1 --port 7862
```

浏览器：

`http://127.0.0.1:7862`

## 10. Smoke Test 结果

使用独立测试 DB 和独立测试 export：

- smoke DB: `{smoke_result['smoke_db']}`
- smoke live CSV: `{smoke_result['smoke_live_csv']}`
- smoke export CSV: `{smoke_result['smoke_export_csv']}`
- tested label_hash: `{smoke_result['tested_label_hash']}`
- action: `{smoke_result['action']}`
- chosen_index: `{smoke_result['chosen_index']}`
- groups_page_count: `{smoke_result['groups_page_count']}`
- progress_before: `{json.dumps(smoke_result['progress_before'], ensure_ascii=False)}`
- progress_after: `{json.dumps(smoke_result['progress_after'], ensure_ascii=False)}`
- export_result: `{json.dumps(smoke_result['export_result'], ensure_ascii=False)}`

## 11. 警告与限制

- 网站依赖本地文件路径服务 `manual_hq_review/images/` 与 `qwen_vl_hq_review_full/panels/`
- 当前站点不生成 pair manifest
- 当前站点不写回原人工审核网站
- `P0/P1/P2/P3` 是启发式风险分层，不等同于最终训练可用性

## 12. 下一步建议

1. 先优先处理 `P0 + pending`
2. 再处理 `P1 + pending`
3. 复核完成后导出 `qwen_full_audit_export.csv`
4. 后续如需构造 final pair manifest，请基于导出 CSV 另写独立脚本执行
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    app = load_app_module()
    ns = make_namespace(args)

    site = app.create_site(ns)
    site.export_current_state()
    progress = site.get_progress()
    smoke_result = app.run_smoke_test(ns)
    write_report(REPORT_PATH, args, progress, smoke_result)

    print("Qwen full audit site completed.")
    print(f"Site dir: {SITE_DIR}")
    print(f"App: {APP_PATH}")
    print(f"SQLite DB: {args.db_path.resolve()}")
    print(f"Live CSV: {args.live_csv.resolve()}")
    print(f"Export CSV: {args.export_csv.resolve()}")
    print(f"Report: {REPORT_PATH}")
    print("No original manual review site state, Qwen full result, OCR source/data/config/checkpoint/output, or original dataset files were modified.")


if __name__ == "__main__":
    main()
