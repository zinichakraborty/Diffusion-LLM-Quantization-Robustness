import os


def set_huggingface_env_vars():
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
