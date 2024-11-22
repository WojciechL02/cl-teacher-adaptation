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
    PLOT_LINEWIDTH,
    TEXT_FONTSIZE,
    TICK_FONTSIZE,
    NAME_LIN,
)
from tqdm import tqdm
from wandb import Api

sns.set_style("darkgrid")


def parse_run_sg(run, num_tasks):
    seed = run.config["seed"]
    run_name = run.group

    metric_name = "cont_eval/sg_normal_avg"
    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    return [
        {
            "run_name": run_name,
            "seed": seed,
            "task": step,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval, 2)
    ]


def parse_run(run, num_tasks, metric_name):
    seed = run.config["seed"]
    run_name = run.group

    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    max_steps = len(cont_eval)
    steps_per_task = max_steps // num_tasks
    return [
        {
            "run_name": run_name,
            "seed": seed,
            "task": step / steps_per_task,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval)
    ]


def plot_sg(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
    num_exemplars = 2000
    approaches = ["ft_nmc", "finetuning"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": tag,
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.num_exemplars": num_exemplars,
            "config.approach": {"$in": approaches},
            "state": "finished",
        },
    )
    runs_ = list(runs)

    runs = []
    for r in runs_:
        if "no_mem_learning" not in r.group and "full_set_prot" not in r.group and "wu" not in r.group:
            runs.append(r)

    # Parse runs to plotting format
    parsed_runs = [
        parse_run_sg(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_finetuning_t10s10_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "cifar100_icarl_ft_nmc_t10s10_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"Stability Gap"
    yticks = [10, 20, 30, 40, 50, 60, 70]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax,
        legend=False
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(2, num_tasks+1))
    plot.set_xlim(2, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)


def plot_wc(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
    num_exemplars = 2000
    approaches = ["ft_nmc", "finetuning"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": tag,
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.num_exemplars": num_exemplars,
            "config.approach": {"$in": approaches},
            "state": "finished",
        },
    )
    runs_ = list(runs)

    runs = []
    for r in runs_:
        if "no_mem_learning" not in r.group and "full_set_prot" not in r.group and "wu" not in r.group:
            runs.append(r)

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks, metric_name="cont_eval/wc_acc")
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_finetuning_t10s10_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "cifar100_icarl_ft_nmc_t10s10_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"WC-ACC"
    yticks = [10, 20, 30, 40, 50, 60]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax,
        legend=False
    )

    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(num_tasks + 1))
    plot.set_xlim(0, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)


def plot_min(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
    num_exemplars = 2000
    approaches = ["ft_nmc", "finetuning"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": tag,
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.num_exemplars": num_exemplars,
            "config.approach": {"$in": approaches},
            "state": "finished",
        },
    )
    runs_ = list(runs)

    runs = []
    for r in runs_:
        if "no_mem_learning" not in r.group and "full_set_prot" not in r.group and "wu" not in r.group:
            runs.append(r)

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks, metric_name="cont_eval/min_acc")
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_finetuning_t10s10_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "cifar100_icarl_ft_nmc_t10s10_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"min-ACC"
    yticks = [10, 20, 30, 40, 50, 60, 70]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax,
        legend=True
    )

    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(num_tasks + 1))
    plot.set_xlim(0, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)

    handles, labels = plot.get_legend_handles_labels()
    handles = [
        handles[labels.index(NAME_NMC_EX)],
        handles[labels.index(NAME_LIN)],
    ]
    plot.legend(
        handles=handles,
        labels=["NMC", "Linear"],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        title=None,
    )


def main():
    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 3, figsize=(25.60, 4.80))  # , width_ratios=[1, 1, 1, 1]

    os.environ['WANDB_API_KEY'] = '434fcc1957118a52a224c4d4a88db52186983f58'

    plot_sg(axes[0], xlabel="Finished Task", ylabel=None)
    plot_wc(axes[1], xlabel="Finished Task", ylabel=None)
    plot_min(axes[2], xlabel="Finished Task", ylabel=None)

    output_path_png = output_dir / "apx_merged_sg.png"
    output_path_pdf = output_dir / "apx_merged_sg.pdf"
    plt.tight_layout()
    plt.savefig(str(output_path_png))
    plt.savefig(str(output_path_pdf))


if __name__ == "__main__":
    main()
