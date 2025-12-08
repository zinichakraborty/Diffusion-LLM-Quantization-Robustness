import argparse
import re
import subprocess
from pathlib import Path

from CoDA.evaluation.utils.qwen_latency_utils import (
    JOB_NAME_TEMPLATE,
    LOG_FILE_PATH_TEMPLATE,
    MEM,
    GRES_GPU,
    TIME,
    SCRIPT_TEMPLATE,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=200,
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


def safe_model_name(model_path: Path):
    return str(model_path.name).replace("/", "__")


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


def create_jobs(models_dir: Path, num_runs: int, warmup_runs: int):
    model_dirs = find_model_directories(models_dir)
    job_ids = []

    for model_dir in model_dirs:

        safe_name = safe_model_name(model_dir)
        model_dir_abs = model_dir.resolve()

        script = SCRIPT_TEMPLATE.format(
            model_dir=str(model_dir_abs),
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        ).strip()

        job_name = JOB_NAME_TEMPLATE.format(model=safe_name).strip()
        log_output = LOG_FILE_PATH_TEMPLATE.format(model=safe_name).strip()

        script_dir = model_dir_abs

        job_id = submit_job(script, job_name, log_output, script_dir)
        job_ids.append(job_id)
        print(
            f"Submitted job {job_id} for latency evaluation on {model_dir.name}"
        )

    return job_ids


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    create_jobs(models_dir, args.num_runs, args.warmup_runs)
    print("All jobs submitted successfully")


if __name__ == "__main__":
    main()
