import argparse

from MoaiPipelineManager import Manager
from run_export import parse_opt as parse_export_opt, main as run_export_main

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project", default="20250115")
    parser.add_argument("--subproject", default="test_sub")
    parser.add_argument("--task", default="test_task")
    parser.add_argument("--version", default="v1")

    moai_args, _ = parser.parse_known_args()
    return moai_args

def setup_export_config(manager: Manager):
    weights = manager.get_best_weight_path()
    hyp = manager.get_train_result_hyp_yaml()
    imgsz = hyp["imgsz"]
    batch_size = 1
    opset = 16

    return weights, imgsz, batch_size, opset

def main():
    args = parse_args()
    manager = Manager(**vars(args))

    weights, imgsz, batch_size, opset = setup_export_config(manager)
    
    opt = parse_export_opt(known=True)

    # Update opt with our parameters
    params = {
        "weights": weights,
        "imgsz": [imgsz, imgsz],
        "batch_size": batch_size,
        "opset": opset,
    }

    for k, v in params.items():
        setattr(opt, k, v)

    # Run export
    run_export_main(opt)

if __name__ == "__main__":
    main()