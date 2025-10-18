import argparse
from pathlib import Path
from utils import EVALUATION_SETS, SCRIPT_TEMPLATE, JOB_NAME_TEMPLATE, LOG_FILE_PATH_TEMPLATE, MEM, GRES_GPU, TIME, BATCH_SIZE
import pyslurm

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

def submit_job(script, job_name, log_output):
    desc = pyslurm.JobSubmitDescription(
        name=job_name,
        time_limit=TIME,
        nodes=1,
        memory_per_node=MEM,
        gres_per_node=GRES_GPU,
        standard_output=log_output,
        standard_error=log_output,
        script=script,
        working_directory=str(Path.cwd())
    )
    job_id = desc.submit()
    return job_id

def create_jobs(model, results_dir):
    job_ids = []
    for evaluation_set in EVALUATION_SETS:
        script = SCRIPT_TEMPLATE.format(model=model, results_dir=results_dir, evaluation_set=evaluation_set, batch_size=BATCH_SIZE).strip()
        job_name = JOB_NAME_TEMPLATE.format(evaluation_set=evaluation_set, model=model).strip()
        log_output = LOG_FILE_PATH_TEMPLATE.format(evaluation_set=evaluation_set, model=model.replace("/", "_")).strip()
        job_id = submit_job(script, job_name, log_output)
        job_ids.append(job_id)
        print(f"Submitted job {job_id} for {evaluation_set} evaluation on {model}")

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