
from pathlib import Path
from src.experiment.experiment import Experiment
from src.experiment.exp_params import ExpParams
from src.utils.device import get_best_device
from src.utils.logger import create_logger
from typing import Literal

import typer

app = typer.Typer()

@app.command()
def run(
    config_path: Path = Path("configs/default.json"),
    job_name: str = "",
    device_str: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    mode: Literal["train", "evaluate", "visualize"] = "train"
) -> None:
    """ Run experiment with given configuration."""

    params = ExpParams.from_json(config_path)
    run_id = job_name or params.generate_run_id()
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(run_dir)
    device = get_best_device(device_str=device_str, 
                             logger=logger)

    experiment = Experiment(
        params=params,
        run_dir=run_dir,
        device=device,
        logger=logger,
    )

    match mode:
        case "train":
            experiment.train()
        case "evaluate":
            experiment.evaluate()
        case "visualize":
            experiment.visualize()
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'train', 'evaluate', or 'visualize'.")


if __name__ == "__main__":
    app()
