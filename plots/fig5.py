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
    metric_name = "cont_eval_acc_tag/t_0"
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


def main():
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path_png = output_dir / "fig5b.png"
    output_path_pdf = output_dir / "fig5b.pdf"

    # Filters for the runs
    tag = "slca"
    dataset = "aircrafts"
    num_tasks = 10
    nepochs = 100
    approaches = ["ft_nmc", "finetuning"]
    names = ["zany-elevator-1470", "electric-capybara-1465", "x0.1", "helpful-fire-1469", "astral-snowflake-1466"]
    air_names = ["x0.1", "no-slca"]

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
            "state": "finished",
        },
    )
    runs = list(runs)
    runs = [r for r in runs if r.name not in air_names]

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
        "aircrafts_finetuning_t10s20_hz_m:10": NAME_FT,
        "aircrafts_ft_nmc_t10s20_hz_m:10_up:1": NAME_NMC_EX,
    }
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot
    plt.figure()
    plt.clf()
    plt.cla()

    # Plot configuration
    xlabel = "Task"
    ylabel = "Task 1 Accuracy"
    title = "Aircrafts | 10 tasks"
    yticks = [20, 40, 60, 80, 100]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plt.xticks(range(num_tasks + 1))
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

    # Remove legend title and set fontsize
    # Reorder labels and handles for the legneds
    handles, labels = plot.get_legend_handles_labels()
    handles = [
        handles[labels.index(NAME_FT)],
        handles[labels.index(NAME_NMC_EX)],
    ]
    plot.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        title=None,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path_png))
    plt.savefig(str(output_path_pdf))


if __name__ == "__main__":
    main()
