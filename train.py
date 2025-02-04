import os
import argparse
import shutil
import torch
import yaml

from MoaiPipelineManager import Manager

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project", default="20250115")
    parser.add_argument("--subproject", default="test_sub")
    parser.add_argument("--task", default="segment_test")
    parser.add_argument("--version", default="v3")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    manager = Manager(**vars(args))

    hyp_path = manager.get_hyp_yaml_path()
    data_path = manager.get_data_yaml_path()
    with open(data_path, "r+") as f:
        data = yaml.safe_load(f)
        data["train"] = f"{manager.get_train_dataset_path()}/train/images"
        data["val"] = f"{manager.get_train_dataset_path()}/valid/images"

    os.remove(data_path)
    with open(data_path, "w") as f:
        yaml.dump(data, f)

    hyp = manager.get_hyp_yaml()
    imgsz = hyp["imgsz"]
    batch_size = hyp["batch_size"]
    epochs = hyp["epochs"]
    resume = hyp["resume"]
    device = '0' if torch.cuda.is_available() else 'cpu'

    weight_type = hyp["weights"]
    weights = f"{os.getcwd()}/weights/yolov5{weight_type}.pt"

    execute_file = "run_seg_train.py" if weight_type == "m_seg" else "run_train.py"

    ocmd = f"python {execute_file} \
    --imgsz {imgsz} \
    --batch {batch_size} \
    --epochs {epochs} \
    --data {data_path} \
    --hyp {hyp_path} \
    --weights {weights} \
    --device {device} \
    --project {manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version} \
    --name training_result \
    --optimizer AdamW \
    --patience {epochs} \
    "

    if resume:
        ocmd += f"--resume "

    os.system(ocmd)

    # weights 폴더 이동
    src = f"{manager.get_training_result_folder_path()}/weights"
    dst = manager.get_weight_folder_path()
    shutil.move(src, dst)

if __name__ == "__main__":
    main()