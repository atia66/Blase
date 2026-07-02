import os
import json
from pathlib import Path

METRICS_DIR = Path(os.environ.get("BLASE_METRICS_DIR", ".")) / "matrics"


def _file() -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return METRICS_DIR / "data.json"


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def record_request(
    worker_name: str,
    *,
    metrics: dict | None = None,
) -> None:
    path = _file()
    data = _read(path)

    existing = data.get(worker_name)
    if not existing:
        data[worker_name] = metrics
    else:
        # merge each key (e.g. "mcq", "choices", "essays") by extending lists
        for key, value in metrics.items():
            if key not in existing:
                existing[key] = value
                continue

            old_val = existing[key]

            if isinstance(value, list) and isinstance(old_val, list):
                old_val.extend(value)

            elif isinstance(value, dict) and isinstance(old_val, dict):
                # e.g. {"ground_truth": [...], "predicted": [...]}
                for sub_key, sub_val in value.items():
                    if sub_key in old_val and isinstance(old_val[sub_key], list):
                        old_val[sub_key].extend(sub_val)
                    else:
                        old_val[sub_key] = sub_val

        data[worker_name] = existing

    _write(path, data)


ground_truth_dir = "./ground_truth/ground_truth"
predicted_dir    = "./ground_truth/predicted"

for folder_name in os.listdir(ground_truth_dir):
    gt_folder = os.path.join(ground_truth_dir, folder_name)
    pt_folder = os.path.join(predicted_dir, folder_name)

    if not os.path.isdir(pt_folder):
        continue

    common_files = set(os.listdir(gt_folder)) & set(os.listdir(pt_folder))
    for file in common_files:
        gt_path = os.path.join(gt_folder, file)
        pt_path = os.path.join(pt_folder, file)
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        with open(pt_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        if folder_name == "answer":
            mcq = [
                {"predicted": pred, "ground_truth": gt_data["mcq"][idx]}
                for idx, pred in enumerate(pred_data["mcq"])
            ]
            choices = [
                {"ground_truth": gt, "predicted": pred_data["choices"][idx]}
                for idx, gt in enumerate(gt_data["choices"])
            ]
            # essays = [
            #     {"ground_truth": gt, "predicted": pred_data["essays"][idx][0]}
            #     for idx, gt in enumerate(gt_data["essays"])
            # ]

            record_request("recognition",metrics={"mcq":mcq,"choices": choices})
        
        else:
            print(gt_path)
            mcq={"ground_truth": [], "predicted":[] }
            for idx, key in enumerate(gt_data["mcq"].keys()):
            
                    # mcq["ground_truth"].append(1 if gt["is_correct"]==True else 0)
                    mcq["ground_truth"].append(gt_data["mcq"][key])
                    # mcq["predicted"].append(1 if gt["is_correct"]==True else 0)
                    mcq["predicted"].append(pred_data["mcq"][key])

            
            complete={"ground_truth": [], "predicted":[] }
            for idx, key in enumerate(gt_data["complete"].keys()):
            
                    # complete["ground_truth"].append(1 if gt["is_correct"]==True else 0)
                    complete["ground_truth"].append(gt_data["complete"][key])
                    # complete["predicted"].append(1 if gt["is_correct"]==True else 0)
                    complete["predicted"].append(pred_data["complete"][key])
            written={"ground_truth": [], "predicted":[] }
            for idx, key in enumerate(gt_data["written"].keys()):

                    written["ground_truth"].append( gt_data["written"][key]["overall_score"])
                    # written["predicted"].append( gt["raw_score"])

                    written["predicted"].append(pred_data["written"][key]["overall_score"])
            record_request("grading",metrics={"mcq":mcq,"choices": complete,"essays":  written})
            

