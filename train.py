import os
import argparse
import shutil
import torch
import yaml

from run_train import parse_opt, main as run_train_main
from MoaiPipelineManager import Manager
from utils.callbacks import Callbacks
from custom_callbacks import (
    train_ready_callback,
    train_start_callback,
    train_epoch_end_callback,
    train_end_callback
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="20250115")
    parser.add_argument("--subproject", type=str, default="test_sub")
    parser.add_argument("--task", type=str, default="test_task")
    parser.add_argument("--version", type=str, default="v3")

    moai_args, remaining_args = parser.parse_known_args()

    opt = parse_opt(known=True)

    for k, v in vars(moai_args).items():
        setattr(opt, k, v)

    return opt, moai_args

def setup_training_config(manager: Manager):
    hyp_path = manager.get_hyp_yaml_path()
    unused_augment = ['degrees', 'translate', 'shear', 'perspective', 'Tmosaic', 'mixup', 'copy_paste']
    hyp = manager.get_hyp_yaml()
    for k in unused_augment:
        hyp[k] = 0

    with open(hyp_path, "w") as f:
        yaml.dump(hyp, f)

    data_path = manager.get_data_yaml_path()
    with open(data_path, "r+") as f:
        data = yaml.safe_load(f)
        data["train"] = f"{manager.get_train_dataset_path()}/train/images"
        data["val"] = f"{manager.get_train_dataset_path()}/valid/images"

    os.remove(data_path)
    with open(data_path, "w") as f:
        yaml.dump(data, f)
    
    return hyp, hyp_path, data_path

def main():
    opt, args = parse_args()
    manager = Manager(**vars(args))

    hyp, hyp_path, data_path = setup_training_config(manager)
    params = {
        "imgsz"     : hyp.get("imgsz"),
        "batch_size": hyp.get("batch_size"),
        "epochs"    : hyp.get("epochs"),
        "patience"  : hyp.get("epochs"),
        "weights"   : f"{os.getcwd()}/weights/yolov5{hyp.get('weights')}.pt",
        "device"    : '0' if torch.cuda.is_available() else 'cpu',
        "data"      : data_path,
        "hyp"       : hyp_path,
        "project"   : f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}",
        "name"      : "training_result",
        "optimizer" : "AdamW",
        "resume"    : False
    }

    for k, v in params.items():
        setattr(opt, k, v)

    callbacks = Callbacks()
    callbacks.register_action(
        hook="on_pretrain_routine_start",
        name="train_ready_callback",
        callback=train_ready_callback
    )
    callbacks.register_action(
        hook="on_train_start",
        name="train_start_callback",
        callback=train_start_callback
    )
    callbacks.register_action(
        hook="on_fit_epoch_end",
        name="train_epoch_end_callback",
        callback=train_epoch_end_callback
    )
    callbacks.register_action(
        hook="on_train_end",
        name="train_end_callback",
        callback=train_end_callback
    )

    run_train_main(opt, callbacks)

    # weights 폴더 이동
    src = f"{manager.get_training_result_folder_path()}/weights"
    dst = manager.get_weight_folder_path()
    shutil.move(src, dst)

if __name__ == "__main__":
    main()