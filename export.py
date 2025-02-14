import os
import argparse

from MoaiPipelineManager import Manager

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project", default="20250115")
    parser.add_argument("--subproject", default="test_sub")
    parser.add_argument("--task", default="test_task")
    parser.add_argument("--version", default="v1")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    manager = Manager(**vars(args))

    weights = manager.get_best_weight_path()
    
    hyp = manager.get_train_result_hyp_yaml()
    imgsz = hyp["imgsz"]
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