import argparse
from pathlib import Path
from utils.latency_utils import (
    JOB_NAME_TEMPLATE,
    LOG_FILE_PATH_TEMPLATE,
    MEM,
    GRES_GPU,
    TIME,
    SCRIPT_TEMPLATE,
)
import pyslurm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing GPTQ model subdirectories (e.g., gptq_experiments/gptq_quantized_models)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of forward passes to measure for latency",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of warmup runs before measuring latency",
    )
    return parser.parse_args()

def find_model_directories(models_dir: Path):
    """Find all subdirectories containing model.safetensors"""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory {models_dir} does not exist")
    
    model_dirs = []
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and (subdir / "model.safetensors").exists():
            model_dirs.append(subdir)
    
    if not model_dirs:
        raise ValueError(
            f"No model directories with model.safetensors found in {models_dir}"
        )
    
    return sorted(model_dirs)

def safe_model_name(model_path: Path):
    """Convert model path to safe name for job naming"""
    return str(model_path.name).replace("/", "__")

def submit_job(script, job_name, log_output):
    """Submit a SLURM job"""
    desc = pyslurm.JobSubmitDescription(
        name=job_name,
        time_limit=TIME,
        nodes=1,
        gres_per_node=GRES_GPU,
        # memory_per_node=MEM,
        # gres_per_node=GRES_GPU,
        # gpus=1,
        standard_output=log_output,
        standard_error=log_output,
        script=script,
        working_directory=str(Path.cwd()),
    )
    job_id = desc.submit()
    return job_id

def create_jobs(models_dir: Path, num_runs: int, warmup_runs: int):
    """Create SLURM jobs for all models in the directory"""
    model_dirs = find_model_directories(models_dir)
    job_ids = []
    
    for model_dir in model_dirs:
        if model_dir.name.startswith("Salesforce"):
            continue  # Skip CoDA directory if present
        safe_name = safe_model_name(model_dir)
        # Use absolute path for model directory
        model_dir_abs = model_dir.resolve()
        script = SCRIPT_TEMPLATE.format(
            model_dir=str(model_dir_abs),
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        ).strip()
        
        job_name = JOB_NAME_TEMPLATE.format(model=safe_name).strip()
        log_output = LOG_FILE_PATH_TEMPLATE.format(model=safe_name).strip()
        
        job_id = submit_job(script, job_name, log_output)
        job_ids.append(job_id)
        print(
            f"Submitted job {job_id} for latency evaluation on {model_dir.name}"
        )
    
    return job_ids

def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    
    print(f"Looking for GPTQ models in {models_dir}")
    print("Results will be saved in each model's directory as latency_result_timestamp.json")
    
    create_jobs(models_dir, args.num_runs, args.warmup_runs)
    
    print("All jobs submitted successfully")

if __name__ == "__main__":
    main()

