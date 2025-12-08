import argparse
import re
import subprocess
from pathlib import Path

from utils.qwen_eval_utils import (
    EVALUATION_SETS,
    SCRIPT_TEMPLATE,
    JOB_NAME_TEMPLATE,
    LOG_FILE_PATH_TEMPLATE,
    MEM,
    GRES_GPU,
    TIME,
    BATCH_SIZE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
    )
    return parser.parse_args()


def find_model_directories(models_dir: Path):
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError()

    model_dirs = []
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and (subdir / "model.safetensors").exists():
            model_dirs.append(subdir)

    return sorted(model_dirs)


def safe_model_name(model_path_or_str):
    return str(model_path_or_str).replace("/", "__")


def create_results_root_dir(results_dir: str | Path) -> Path:
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def submit_job(script: str, job_name: str, log_output: str, script_dir: Path):
    script_dir = Path(script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    script_path = script_dir / f"{job_name}.sh"
    script_path.write_text(script)

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--time={TIME}",
        f"--gres={GRES_GPU}",
        f"--output={log_output}",
        f"--error={log_output}",
    ]

    if MEM:
        cmd.append(f"--mem={MEM}")

    cmd.append(str(script_path))

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.strip()
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise RuntimeError()

    job_id = int(match.group(1))
    return job_id


def create_jobs(models_dir: Path, results_root: Path):
    model_dirs = find_model_directories(models_dir)
    model_dirs = [d for d in model_dirs if "Qwen3" in d.name]
    all_job_ids = []

    for model_dir in model_dirs:

        model_dir_abs = model_dir.resolve()
        safe_name = safe_model_name(model_dir_abs.name)

        model_results_dir = results_root / safe_name
        model_results_dir.mkdir(parents=True, exist_ok=True)

        for evaluation_set in EVALUATION_SETS:
            script = SCRIPT_TEMPLATE.format(
                model=str(model_dir_abs),
                safe_model_name=safe_name,
                results_dir=str(model_results_dir),
                evaluation_set=evaluation_set,
                batch_size=BATCH_SIZE,
            ).strip()

            job_name = JOB_NAME_TEMPLATE.format(
                evaluation_set=evaluation_set,
                model=safe_name,
            ).strip()

            log_output = LOG_FILE_PATH_TEMPLATE.format(
                evaluation_set=evaluation_set,
                model=safe_name,
            ).strip()

            script_dir = model_dir_abs

            job_id = submit_job(script, job_name, log_output, script_dir)
            all_job_ids.append(job_id)
            print(
                f"Submitted job {job_id} for {evaluation_set} evaluation on {model_dir.name}"
            )

    return all_job_ids


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    results_root = create_results_root_dir(args.results_dir)
    create_jobs(models_dir, results_root)
    print("All jobs submitted successfully")


if __name__ == "__main__":
    main()
