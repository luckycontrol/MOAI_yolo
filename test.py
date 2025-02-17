import os
import sys
import yaml
import argparse
import shutil
import torch
import glob

from MoaiPipelineManager import Manager
from run_test import parse_opt as parse_test_opt, main as run_test_main
from run_seg_test import parse_opt as parse_seg_test_opt, main as run_seg_test_main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="20250115")
    parser.add_argument("--subproject", type=str, default="test_sub")
    parser.add_argument("--task", type=str, default="test_task")
    parser.add_argument("--version", type=str, default="v1")

    moai_args, _ = parser.parse_known_args()
    return moai_args

def setup_testing_config(manager: Manager):
    opt_path = f"{manager.get_training_result_folder_path()}/opt.yaml"
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)

    source = manager.get_test_dataset_path()
    imgsz = opt["imgsz"]
    weight_type = opt["hyp"]["weights"]
    weights = manager.get_best_weight_path()
    device = '0' if torch.cuda.is_available() else 'cpu'

    return source, imgsz, weight_type, weights, device

def organize_test_results(test_result_folder_path: str):
    txt_files = glob.glob(f"{test_result_folder_path}/labels/*.txt")
    for txt_file in txt_files:
        txt_file_name = os.path.basename(txt_file)
        shutil.move(txt_file, f"{test_result_folder_path}/{txt_file_name}")
    shutil.rmtree(f"{test_result_folder_path}/labels")

def main():
    args = parse_args()

    sys.argv = [sys.argv[0]]

    manager = Manager(**vars(args))

    source, imgsz, weight_type, weights, device = setup_testing_config(manager)
    
    if weight_type == "m_seg":
        opt = parse_seg_test_opt()
    else:
        opt = parse_test_opt()

    # Update opt with our parameters
    params = {
        "source": source,
        "imgsz": [imgsz, imgsz],
        "weights": weights,
        "device": device,
        "project": manager.get_version_folder_path(),
        "name": "inference_result",
        "conf_thres": 0.1,
        "save_txt": True,
        "save_conf": True
    }

    for k, v in params.items():
        setattr(opt, k, v)

    # Run inference
    if weight_type == "m_seg":
        run_seg_test_main(opt)
    else:
        run_test_main(opt)

    # Organize results
    organize_test_results(manager.get_test_result_folder_path())

if __name__ == "__main__":
    main()