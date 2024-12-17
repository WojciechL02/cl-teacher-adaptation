import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import (
    COLOR_PALETTE,
    HUE_ORDER,
    LEGEND_FONTSIZE,
    NAME_FT,
    NAME_NMC_EX,
    NAME_NMC_FULL,
    NAME_JOINT,
    NAME_JOINT_NMC,
    NAME_SCR,
    PLOT_LINEWIDTH,
    TEXT_FONTSIZE,
    TICK_FONTSIZE,
)
from tqdm import tqdm
from wandb import Api

sns.set_style("darkgrid")


def parse_run(run, num_tasks):
    seed = run.config["seed"]
    run_name = run.group

    # download all the values of 'cont_eval_acc_tag/t_0' from the run
    metric_name = "cont_eval/task_recency_bias"
    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    max_steps = len(cont_eval)
    steps_per_task = max_steps // (num_tasks-1)
    return [
        {
            "run_name": run_name,
            "seed": seed,
            "task": step / steps_per_task + 1,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval)
    ]


def main():
    os.environ['WANDB_API_KEY'] = '434fcc1957118a52a224c4d4a88db52186983f58'
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path_png = output_dir / "fig_scr_ltb_in.png"
    output_path_pdf = output_dir / "fig_scr_ltb_in.pdf"

    # Filters for the runs
    tag = ["scr_thesis",]
    dataset = "imagenet_subset_kaggle"
    num_tasks = 10
    nepochs = 100
    approaches = ["finetuning", "scr"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": {"$in": tag},
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.approach": {"$in": approaches},
            "state": "finished",
        },
    )
    runs = list(runs)
    # runs = [r for r in runs if "figure1" in r.tags or "joint" in r.tags]
    runs = [r for r in runs if r.config["num_exemplars"] == 2000]
    groups = ["imagenet_subset_kaggle_finetuning_t10s10_hz_m:2000", "imagenet_subset_kaggle_scr_t10s10_hz_m:2000"]
    runs = [r for r in runs if r.group in groups]
    for r in runs:
        if r.group == "imagenet_subset_kaggle_finetuning_t10s10_hz_m:2000" and r.config["classifier"] == "nmc":
            r.group += "_nmc"
    print(len(runs))

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "imagenet_subset_kaggle_finetuning_t10s10_hz_m:2000": NAME_FT,
        "imagenet_subset_kaggle_finetuning_t10s10_hz_m:2000_nmc": NAME_NMC_EX,
        "imagenet_subset_kaggle_scr_t10s10_hz_m:2000": NAME_SCR,
    }
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot
    plt.figure()
    plt.clf()
    plt.cla()

    # Plot configuration
    xlabel = "Finished Task"
    ylabel = "Latest Task Prediction Bias"
    title = "ImageNet100 | 10 tasks"
    yticks = [0.2, 0.4, 0.6, 0.8]

    hue_order = {
        NAME_FT: 1,
        NAME_NMC_EX: 2,
        NAME_SCR: 3,
    }

    # dashes_dict = {
    #     NAME_FT: (2, 0),
    #     NAME_NMC_EX: (2, 0),
    #     NAME_JOINT: (3, 3),
    #     NAME_JOINT_NMC: (3, 3),
    #     NAME_NMC_FULL: (2, 0),
    # }

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=hue_order,
        # style="run_name",
        # dashes=dashes_dict,
        linewidth=PLOT_LINEWIDTH,
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plt.xticks(range(1, num_tasks+1))
    plot.set_xlim(1, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)

    # Remove legend title and set fontsize
    # Reorder labels and handles for the legneds
    handles, labels = plot.get_legend_handles_labels()
    handles = [
        handles[labels.index(NAME_FT)],
        handles[labels.index(NAME_NMC_EX)],
        handles[labels.index(NAME_SCR)],
    ]
    plot.legend(
        handles=handles,
        labels=["Finetuning", "FT (NMC)", "SCR"],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        title=None,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path_png), bbox_inches='tight', pad_inches=0)
    plt.savefig(str(output_path_pdf), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
