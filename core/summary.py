import re
import os
import glob
from pickle import load
from numpy import nanmean
from argparse import ArgumentParser


levels = {
    "Easy": {
        "min_difficulty": 0.0,
        "max_difficulty": 0.35,
    },
    "Medium": {
        "min_difficulty": 0.35,
        "max_difficulty": 0.45,
    },
    "Hard": {
        "min_difficulty": 0.45,
        "max_difficulty": 0.5,
    },
}


def score_success_with_tolerance(score, pos_tol=0.02, ori_tol=0.1):
    pos_all = score["debug"].get("positions_reached", [])
    ori_all = score["debug"].get("orientations_reached", [])

    if not pos_all or not ori_all:
        return score["success"]

    if all(pos_all) and all(ori_all):
        return score["success"]

    for i, (pos_ok, ori_ok) in enumerate(zip(pos_all, ori_all)):
        pos_res = score["debug"]["position_residuals"][i]
        ori_res = score["debug"]["orientation_residuals"][i]

        pos_check = pos_ok or pos_res < pos_tol + 1e-2
        ori_check = ori_ok or ori_res < ori_tol

        if not (pos_check and ori_check):
            return 0
    return 1


def print_summary(scores):
    total_levels = len(levels)
    total_evals = args.eval_tests * total_levels

    def scaled_mean(data, denominator):
        return nanmean(data) * len(data) / denominator if data else None

    header_cols = (
        ["# Arms"]
        + list(levels.keys())
        + [
            "Avg (Count)",
            "Collision Failure Rate",
        ]
    )

    rows = []
    rows.append(header_cols)

    agents = sorted({score["task"]["arm_count"] for score in scores})
    for arm_count in agents:
        arm_scores = [
            score for score in scores if score["task"]["arm_count"] == arm_count
        ]
        row = [str(arm_count)]
        for lvl in levels.values():
            filtered = [
                score_success_with_tolerance(score)
                for score in arm_scores
                if lvl["min_difficulty"]
                <= score["task"]["difficulty"]
                <= lvl["max_difficulty"]
            ]
            if not filtered:
                row.append("-")
            else:
                val = scaled_mean(filtered, args.eval_tests)
                row.append(f"{val:.03f}")
        all_success = [score_success_with_tolerance(score) for score in arm_scores]
        avg_val = scaled_mean(all_success, total_evals)
        if avg_val is None:
            row.append("-")
        else:
            row.append(f"{avg_val:.03f} ({total_evals})")
        collisions = [
            1
            for score in arm_scores
            if score["success"] == 0 and score["collisions"] > 0
        ]
        col_val = scaled_mean(collisions, total_evals)
        row.append(f"{col_val:.03f}" if col_val is not None else "-")
        rows.append(row)

    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header_cols))]

    def format_row(row):
        return (
            "| "
            + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    header_line = format_row(rows[0])
    print(header_line)
    sep = "| " + " | ".join("-" * col_widths[i] for i in range(len(header_cols))) + " |"
    print(sep)
    for row in rows[1:]:
        print(format_row(row))


def print_summary_sheet(scores):
    total_levels = len(levels)
    total_evals = args.eval_tests * total_levels

    def scaled_mean(data, denominator):
        return nanmean(data) * len(data) / denominator if data else None

    header_cols = (
        ["# Arms"]
        + list(levels.keys())
        + [
            "Avg (Count)",
            "Collision Failure Rate",
        ]
    )

    rows = []
    rows.append(header_cols)

    agents = sorted({score["task"]["arm_count"] for score in scores})
    for arm_count in agents:
        arm_scores = [
            score for score in scores if score["task"]["arm_count"] == arm_count
        ]
        row = [str(arm_count)]
        for lvl in levels.values():
            filtered = [
                score_success_with_tolerance(score)
                for score in arm_scores
                if lvl["min_difficulty"]
                <= score["task"]["difficulty"]
                <= lvl["max_difficulty"]
            ]
            if not filtered:
                row.append("-")
            else:
                val = scaled_mean(filtered, args.eval_tests)
                row.append(f"{val:.03f}")
        all_success = [score_success_with_tolerance(score) for score in arm_scores]
        avg_val = scaled_mean(all_success, total_evals)
        row.append(f"{avg_val:.03f} ({total_evals})" if avg_val is not None else "-")
        collisions = [
            1
            for score in arm_scores
            if score["success"] == 0 and score["collisions"] > 0
        ]
        col_val = scaled_mean(collisions, total_evals)
        row.append(f"{col_val:.03f}" if col_val is not None else "-")
        rows.append(row)

    # Print rows in TSV format
    for row in rows:
        print(",".join(row))


if __name__ == "__main__":
    parser = ArgumentParser("Benchmark Score Summarizer")
    parser.add_argument(
        "--path",
        type=str,
        default="runs/plain_diffusion/",
        help="directory or file path",
    )
    parser.add_argument(
        "--eval_tests", type=float, default=100, help="Number of evaluation tests"
    )
    parser.add_argument(
        "--file_prefix", type=str, default="plain_diffusion", help="File prefix"
    )
    args = parser.parse_args()

    all_scores = []

    if os.path.isdir(args.path):
        pkl_files = glob.glob(os.path.join(args.path, "*.pkl"))
    else:
        pkl_files = [args.path]

    pattern = re.compile(rf"^{args.file_prefix}_(\d+)_(\w+)\.pkl$")

    for file_path in pkl_files:
        match = pattern.match(os.path.basename(file_path))
        if match:
            try:
                scores = load(open(file_path, "rb"))
                all_scores.extend(scores)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

    if all_scores:
        print("Summary Table:")
        print_summary(all_scores)
        print("\nSummary Sheet (CSV format):")
        print_summary_sheet(all_scores)
    else:
        print("No valid results found.")
