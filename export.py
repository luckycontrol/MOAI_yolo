import os
import argparse
import shutil
import torch

from MoaiPipelineManager import Manager

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project", default="test_project")
    parser.add_argument("--subproject", default="sub_project")
    parser.add_argument("--task", default="detection")
    parser.add_argument("--version", default="v199")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    manager = Manager(**vars(args))

    weights = manager.get_best_weight_path()
    
    hyp = manager.get_hyp_yaml()
    imgsz = [hyp["imgsz"], hyp["imgsz"]]
    batch_size = 1
    opset = 16

    ocmd = f"python run_export.py \
        --weights {weights} \
        --imgsz {imgsz} \
        --batch-size {batch_size} \
        --opset {opset} \
        "

    os.system(ocmd)

if __name__ == "__main__":
    main()