
from pathlib import Path
from experiment.experiment import Experiment
from experiment.exp_params import ExpParams
from utils.device import get_best_device
from utils.logger import create_logger

import typer

app = typer.Typer()

@app.command()
def run(
    config_path: Path = Path("configs/default.json"),
    job_name: str = "",
    device: str = "auto",
    mode: str = "train",
) -> None:
    """ Run experiment with given configuration."""

    params = ExpParams.from_json(config_path)
    run_id = job_name or params.generate_run_id()
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(run_dir)
    device = get_best_device(device_str=device, logger=logger)

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
