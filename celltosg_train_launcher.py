import os
import subprocess
from itertools import product
from datetime import datetime

# Global hyperparameters for the baseline model training
global_hyperparams = {
    # "disease_name": [
    #     "Alzheimer's Disease",
    #     "Parkinson's Disease",
    #     "Cancer"
    # ],
    "train_lr": [0.001, 0.0005, 0.0001],
    "train_batch_size": [16],
    "train_base_layer": [
        "gcn",
        "gat",
        "gin",
        "transformer"
    ],
}

# Model-specific hyperparameters
model_specific_defaults = {
    "gcn": {
        "train_batch_size": [8],
    },
    "gat": {
        "train_batch_size": [4]
    },
    "gin": {
        "train_batch_size": [8],
    },
    "transformer": {
        "train_batch_size": [8],
    },

}

# Log
log_dir = "log/celltosg_train"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "run_log.txt")

def log(msg: str):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} {msg}\n")


for model in global_hyperparams["train_base_layer"]:
    local_hyperparams = dict(global_hyperparams)
    local_hyperparams["train_base_layer"] = [model]
    if model in model_specific_defaults:
        for key, value in model_specific_defaults[model].items():
            local_hyperparams[key] = value
    keys = list(local_hyperparams.keys())
    values = list(local_hyperparams.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        run_name = "_".join(str(params[k]).replace("'", "").replace(" ", "_") for k in keys)

        cmd = ["python", "train.py"]
        for k, v in params.items():
            cmd.append(f"--{k}={v}")

        # Separate log file for each run
        run_log_path = os.path.join(log_dir, f"{run_name}.log")

        print(f"▶ Running: {run_name}")
        print("CMD:", " ".join(cmd))

        try:
            log(f"▶ Starting run: {run_name}")
            with open(run_log_path, "w") as logfile:
                subprocess.run(cmd, check=True, stdout=logfile, stderr=logfile)
            log(f"✅ Success: {run_name}")
        except subprocess.CalledProcessError:
            log(f"❌ Failed: {run_name}")
