import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import (
    COLOR_PALETTE,
    HUE_ORDER,
    LEGEND_FONTSIZE,
    NAME_NMC_EX,
    NAME_LIN,
    PLOT_LINEWIDTH,
    TEXT_FONTSIZE,
    TICK_FONTSIZE,
)
from tqdm import tqdm
from wandb import Api

sns.set_style("darkgrid")


def parse_run_sg(run, num_tasks):
    seed = run.config["seed"]
    run_name = run.group
    clf = run.config["classifier"]

    metric_name = "cont_eval/sg_normal_avg"
    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    return [
        {
            "run_name": f"{run_name}_{clf}",
            "seed": seed,
            "task": step,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval, 2)
    ]


def parse_run(run, num_tasks, metric_name):
    seed = run.config["seed"]
    run_name = run.group
    clf = run.config["classifier"]

    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    max_steps = len(cont_eval)
    steps_per_task = max_steps // (num_tasks - 1)
    return [
        {
            "run_name": f"{run_name}_{clf}",
            "seed": seed,
            "task": step / steps_per_task + 1,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval)
    ]


def plot_ewc_sg(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "nmc_vs_lin"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
    approaches = ["ewc",]
    clfs = ["nmc", "linear"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": tag,
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.approach": {"$in": approaches},
            "config.classifier": {"$in": clfs},
            "state": "finished",
        },
    )
    runs = list(runs)
    print(len(runs))

    # Parse runs to plotting format
    parsed_runs = [
        parse_run_sg(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_ewc_cifar100t10s10_nmc_vs_lin_hz_nmc": NAME_NMC_EX,
        "cifar100_icarl_ewc_cifar100t10s10_nmc_vs_lin_hz_linear": NAME_LIN,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"Stability Gap"
    yticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette={
            NAME_NMC_EX: "tab:blue",
            NAME_LIN: "tab:red",
        },
        hue_order={
            NAME_NMC_EX: 1,
            NAME_LIN: 2,
        },
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


def plot_ewc(ax, xlabel, ylabel, metric, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "nmc_vs_lin"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
    approaches = ["ewc",]
    clfs = ["nmc", "linear"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": tag,
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.approach": {"$in": approaches},
            "config.classifier": {"$in": clfs},
            "state": "finished",
        },
    )
    runs = list(runs)
    print(len(runs))

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks, metric_name=metric)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_ewc_cifar100t10s10_nmc_vs_lin_hz_nmc": NAME_NMC_EX,
        "cifar100_icarl_ewc_cifar100t10s10_nmc_vs_lin_hz_linear": NAME_LIN,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    if "wc_acc" in metric:
        title = "WC-ACC"
    elif "min_acc" in metric:
        title = "min-ACC"
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
        legend=legend
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(1, num_tasks + 1))
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
    if legend:
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

    plot_ewc_sg(axes[0], xlabel="Finished Task", ylabel=None)
    plot_ewc(axes[1], xlabel="Finished Task", ylabel=None, metric="cont_eval/wc_acc")
    plot_ewc(axes[2], xlabel='Finished Task', ylabel=None, metric="cont_eval/min_acc", legend=True)

    output_path_png = output_dir / "apx_merged_ewc_sg.png"
    output_path_pdf = output_dir / "apx_merged_ewc_sg.pdf"
    plt.tight_layout()
    plt.savefig(str(output_path_png))
    plt.savefig(str(output_path_pdf))


if __name__ == "__main__":
    main()
