import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

models = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
]
dataset = "BFCL_v3_multiple"
query_to_title = {
    "end_to_end_latency_s.mean": "Average end-to-end latency (s)",
    "time_per_output_token_s.mean": "Average time per output token (s)",
    "time_to_first_token_s.mean": "Average time to first token (s)",
    "output_tokens.mean": "Average output tokens",
}


def draw(args, sglang: Dict, query: str):
    colors = ["#0056b3", "#FF8C00"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    bars = ["no_stag", "use_stag"]
    width = 0.35
    gap = 0.03
    x = np.arange(len(models))

    draw_info = {model: sglang[model][dataset] for model in sglang}
    for i, bar in enumerate(bars):
        values = [draw_info[model][bar][query] for model in models]
        ax.bar(x + i * (width + gap), values, width, color=colors[i], label=bar)
    ax.set_title("SGLang", fontsize=25, pad=20)
    ax.set_xticks(x + width / 2)
    ax.set_ylabel(query_to_title[query], fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 0.9)
    ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0.4, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0.6, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(0, 0.9, 0.2))
    ax.set_xticklabels(
        [model.removesuffix("-Instruct") for model in models],
        rotation=0,
        ha="center",
        fontsize=16,
    )
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(2)]

    plt.subplots_adjust(
        bottom=0.2, top=0.85, left=0.05, right=0.95, wspace=5, hspace=0.5
    )
    fig.legend(
        handles=legend_handles,
        labels=["without structual tag", "with structual tag"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(f"{args.bench_root}/{query}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Draw")
    parser.add_argument(
        "--bench-root",
        type=str,
        help="The bench root path of the result.",
    )
    args = parser.parse_args()
    with open(
        f"{args.bench_root}/sglang/bench.json", mode="r", encoding="utf-8"
    ) as file:
        sglang = json.load(file)
    # draw(args, sglang, "output_tokens.mean")
    # draw(args, sglang, "time_per_output_token_s.mean")
    # draw(args, sglang, "time_to_first_token_s.mean")
    draw(args, sglang, "end_to_end_latency_s.mean")
