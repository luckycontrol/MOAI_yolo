import json

keys = [
    "train/box_loss",
    "train/obj_loss",
    "train/cls_loss",  # train loss
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",  # metrics
    "val/box_loss",
    "val/obj_loss",
    "val/cls_loss",  # val loss
    "x/lr0",
    "x/lr1",
    "x/lr2",
    "time",
]

def train_ready_callback():
    message = {
        "status": "preparing",
        "message": "학습 환경을 준비중입니다.",
    }

    print(json.dumps(message, ensure_ascii=False))

def train_start_callback():
    message = {
        "status": "start",
        "message": "모델 학습을 시작합니다."
    }

    print(json.dumps(message, ensure_ascii=False))

def train_epoch_end_callback(log_vals, epoch, best_fitness, fi, time_left_str):
    full_dict = dict(zip(keys, log_vals))
    desired_metrics = ["metrics/mAP_0.5", "metrics/precision", "metrics/recall"]
    metrics = {k: full_dict[k] for k in desired_metrics if k in full_dict}

    log_data = {
        "epoch": epoch,
        "time": time_left_str,
        "metrics/mAP_0.5": metrics['metrics/mAP_0.5'],
        "precision": metrics['metrics/precision'],
        "recall": metrics['metrics/recall'],
    }

    message = {
        "status": "in_progress",
        "message": log_data
    }

    print(json.dumps(message, ensure_ascii=False))

def train_end_callback(last, best, epoch, results):
    message = {
        "status": "complete",
        "message": "모델 학습이 성공적으로 완료되었습니다."
    }

    print(json.dumps(message, ensure_ascii=False))