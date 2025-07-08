from pathlib import Path

#I've made this file to ensure code compatibility across machines when dealing with paths.


PROJECT_ROOT = Path(__file__).resolve().parents[1]        # TGNNs/
DATASETS_DIR = PROJECT_ROOT / "dataworkspace" / "datasets"
MODELS_DIR   = PROJECT_ROOT / "models"                    # etc.

def dataset_root(name: str) -> Path:
    """
    Return .../DataWorkspace/Datasets/<name>, creating it if it doesn't exist.
    """
    path = DATASETS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path