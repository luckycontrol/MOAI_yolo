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
    
    parser.add_argument("--project", default="20250115")
    parser.add_argument("--subproject", default="test_sub")
    parser.add_argument("--task", default="test_task")
    parser.add_argument("--version", default="v3")

    args = parser.parse_args()

    return args

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

if __name__ == "__main__":
    opt = parse_opt()

    args = parse_args()
    manager = Manager(**vars(args))

    hyp, hyp_path, data_path = setup_training_config(manager)

    opt.imgsz = hyp.get("imgsz")
    opt.batch_size = hyp.get("batch_size")
    opt.epochs = hyp.get("epochs")
    opt.weights = f"{os.getcwd()}/weights/yolov5{hyp.get('weights')}.pt"
    opt.device = '0' if torch.cuda.is_available() else 'cpu'
    opt.data = data_path
    opt.hyp = hyp_path
    opt.project = f"{manager.location}/{manager.project}/{manager.subproject}/{manager.task}/{manager.version}"
    opt.name = "training_result"
    opt.optimizer = "AdamW"
    opt.patience = hyp.get("epochs")

    if hyp.get("resume") is not None and hyp["resume"] == True:
        opt.resume = True

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