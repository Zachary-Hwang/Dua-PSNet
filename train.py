import sys
import os
import argparse
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging
import dataloaders
import models
import performance_metrics
import losses

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, weight):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss = BCE_loss(torch.sigmoid(output), target) + weight * Dice_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    # elif args.dataset == "Mix":
    #     img_path = args.root + "images/*"
    #     input_paths = sorted(glob.glob(img_path))
    #     depth_path = args.root + "masks/*"
    #     target_paths = sorted(glob.glob(depth_path))
    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()

    model = models.DuaPSNet()

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    )

def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
    ) = build(args)

    if not os.path.exists("./Trained models"):
        os.makedirs("./Trained models")

    prev_best_test = None
    train_losses = []
    test_performances = []
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    weight = 0.2
    decay_factor = 0.9 
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss, weight
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            logging.info("Saving best model...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                f"Trained models/Best_DuaPSNet_{args.dataset}_epoch{epoch}.pt",
            )
            prev_best_test = test_measure_mean
        
        train_losses.append(loss)
        test_performances.append(test_measure_mean)
        
    return train_losses, test_performances

def get_args():
    parser = argparse.ArgumentParser(description="Train DuaPSNet on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC", "Mix"])
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-4, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )

    return parser.parse_args()

    
def main():
    logging.basicConfig(
        filename="training.log",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    args = get_args()
    train_losses, test_performances = train(args)

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), test_performances, label="Test Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Performance")
    plt.legend()
    plt.title("Loss and Performance Convergence")
    plt.grid(True)
    plt.savefig("loss_and_performance.png")
    

if __name__ == "__main__":
    main()

    