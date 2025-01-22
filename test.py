import os
import yaml
import argparse
import shutil
import torch
import glob

from MoaiPipelineManager import Manager

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project", default="20250115")
    parser.add_argument("--subproject", default="test_sub")
    parser.add_argument("--task", default="segment_test")
    parser.add_argument("--version", default="v1")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    manager = Manager(**vars(args))

    opt_path = f"{manager.get_training_result_folder_path()}/opt.yaml"
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)

    source = manager.get_test_dataset_path()
    imgsz = opt["imgsz"]
    weight_type = opt["hyp"]["weights"]
    weights = manager.get_best_weight_path()
    device = '0' if torch.cuda.is_available() else 'cpu'

    execute_file = "run_seg_test.py" if weight_type == "m_seg" else "run_test.py"

    ocmd = f"python {execute_file} \
    --source {source} \
    --data {manager.get_data_yaml_path()} \
    --imgsz {imgsz} \
    --weights {weights} \
    --conf-thres 0.1 \
    --project {manager.get_version_folder_path()} \
    --name inference_result \
    --device {device} \
    --save-txt --save-conf"

    os.system(ocmd)

    test_result_folder_path = manager.get_test_result_folder_path()
    txt_files = glob.glob(f"{test_result_folder_path}/labels/*.txt")

    for txt_file in txt_files:
        txt_file_name = os.path.basename(txt_file)
        shutil.move(txt_file, f"{test_result_folder_path}/{txt_file_name}")

    shutil.rmtree(f"{test_result_folder_path}/labels")

if __name__ == "__main__":
    main()