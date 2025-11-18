import argparse
import subprocess
from pathlib import Path
from utils.eval_utils import EVALUATION_SETS, SCRIPT_TEMPLATE, JOB_NAME_TEMPLATE, LOG_FILE_PATH_TEMPLATE, MEM, GRES_GPU, TIME, BATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def create_results_dir(results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def check_model(model):
    if not Path(model).exists() and not model.startswith("Salesforce/"):
        raise FileNotFoundError(f"Model path {model} does not exist or is not a Salesforce model")


def submit_job(script: str, job_name: str, log_output: str):
    """
    Submit a job to Slurm using `sbatch`, passing the job script via stdin.
    """
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--time={TIME}",
        "--nodes=1",
        f"--mem={MEM}",
        f"--gres={GRES_GPU}",
        f"--output={log_output}",
        f"--error={log_output}",
        f"--chdir={Path.cwd()}",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=script,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"sbatch failed for job {job_name} with exit code {e.returncode}:\n"
            f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        )

    # Typical sbatch output: "Submitted batch job 12345"
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"sbatch did not return a job ID for job {job_name}")

    job_id = stdout.split()[-1]
    return job_id


def safe_model_name(model):
    return model.replace("/", "__")


def create_jobs(model, results_dir):
    job_ids = []
    for evaluation_set in EVALUATION_SETS:
        script = SCRIPT_TEMPLATE.format(
            model=model,
            safe_model_name=safe_model_name(model),
            results_dir=results_dir,
            evaluation_set=evaluation_set,
            batch_size=BATCH_SIZE,
        ).strip()

        job_name = JOB_NAME_TEMPLATE.format(
            evaluation_set=evaluation_set,
            model=safe_model_name(model),
        ).strip()

        log_output = LOG_FILE_PATH_TEMPLATE.format(
            evaluation_set=evaluation_set,
            model=safe_model_name(model),
        ).strip()

        job_id = submit_job(script, job_name, log_output)
        job_ids.append(job_id)
        print(f"Submitted job {job_id} for {evaluation_set} evaluation on {model}")

    return job_ids


def main():
    args = parse_args()
    model = args.model
    results_dir = args.results_dir

    check_model(model)
    results_dir = create_results_dir(results_dir)
    print(f"Results will be stored in {results_dir}")

    create_jobs(model, results_dir)

    print("Jobs submitted successfully")


if __name__ == "__main__":
    main()