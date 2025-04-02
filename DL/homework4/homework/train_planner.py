import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "linear_planner",
    transform_pipeline="state_only",
    num_workers = 4,
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")


    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers, transform_pipeline = transform_pipeline)
    val_data = load_data("drive_data/val", shuffle=False, transform_pipeline = transform_pipeline)

    loss_func = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    detmet_train = PlannerMetric()
    detmet_val = PlannerMetric()
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        detmet_train.reset()
        detmet_val.reset()

        model.train()

        for x in train_data:
      
            track_left = x['track_left']
            track_right = x['track_right'] 
            waypoints = x['waypoints']
            waypoints_mask = x['waypoints_mask']

            track_left, track_right, waypoints, waypoints_mask = track_left.to(device), track_right.to(device), waypoints.to(device), waypoints_mask.to(device)
            # print(x.keys())
            # print(track_left.shape)
            # print(track_right.shape)
            # print(waypoints.shape)
            # print(waypoints_mask.shape)
            # print("###")
            # print(waypoints[127])
            # print(waypoints_mask[127])
            
            # import sys
            # sys.exit(0)
            optimizer.zero_grad()
            pred_waypoints = model(track_left, track_right)
            # pred = logits.argmax(dim=1)
         
            loss = loss_func(pred_waypoints, waypoints)
            loss.backward()
            optimizer.step()
            
            # preds = torch.argmax(logits, dim=1)
            detmet_train.add(pred_waypoints, waypoints, waypoints_mask)
            
            global_step += 1

        
        with torch.inference_mode():
            model.eval()

            for x in val_data:
                track_left = x['track_left']
                track_right = x['track_right'] 
                waypoints = x['waypoints']
                waypoints_mask = x['waypoints_mask']

                track_left, track_right, waypoints, waypoints_mask = track_left.to(device), track_right.to(device), waypoints.to(device), waypoints_mask.to(device)
                
                pred_waypoints = model(track_left, track_right)
            
                loss = loss_func(pred_waypoints, waypoints)
                detmet_val.add(pred_waypoints, waypoints, waypoints_mask)
            

        metrics_train = detmet_train.compute()
        metrics_val = detmet_val.compute()

        print("-----------Epoch:",epoch,"--------")
        print("train:",metrics_train)
        print("val:",metrics_val)
  
    save_model(model)

    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

   
    train(**vars(parser.parse_args()))