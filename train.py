import logging
import os
os.path.join('../')
import torch
import torch.nn as nn
import torch.nn.functional as functional
from dataset import MyDataset
from inference_hook import InferenceHook
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from cpu import ConfigArgumentParser, EvalHook, Trainer, save_args, set_random_seed, setup_logger

logger = logging.getLogger(__name__)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, target in test_loader:
            output = model((img, target)).cpu()
            # sum up batch loss
            test_loss += functional.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def parse_args():
    parser = ConfigArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--work-dir", type=str, default="work_dir", metavar="DIR",
                        help="Directory to save checkpoints and logs (default: 'work_dir').")
    parser.add_argument("--dataset-dir", type=str, default="E:\\dataset\\voicebank", metavar="DIR",
                        help="Directory to load dataset (default: E:\\dataset\\voicebank).")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N",
                        help="Input batch size for training (default: 1).")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="Input batch size for test (default: 1000).")
    parser.add_argument("--epochs", type=int, default=1, metavar="N",
                        help="Number of epochs to train (default: 1).")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR",
                        help="Learning rate (default: 1e-3).")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
                        help="Learning rate step gamma (default: 0.7).")
    parser.add_argument("--device", type=str, default="cpu", metavar="D",
                        help="Device to train on (default: 'cpu').")
    parser.add_argument("--seed", type=int, default=-1, metavar="S",
                        help="Random seed, set to negative to randomize everything (default: -1).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Turn on the CUDNN deterministic setting.")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="Interval for logging to console and tensorboard (default: 10).")
    return parser.parse_args()


def build_dataset(dir):
    train_dataset = MyDataset(dir)
    test_dataset = MyDataset(is_training=False)
    return train_dataset, test_dataset


def build_dataloader(args):
    train_dataset, test_dataset = build_dataset(args.dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)
    return train_loader, test_loader


def main():
    # 1. Create an argument parser supporting loading YAML configuration file
    args = parse_args()

    # 2. Basic setup
    setup_logger(output_dir=args.work_dir)
    save_args(args, os.path.join(args.work_dir, "runtime_config.yaml"))
    # If args.seed is negative or None, will use a randomly generated seed
    set_random_seed(args.seed, args.deterministic)
    device = torch.device(args.device)

    # 3. Create data_loader, model, optimizer, lr_scheduler
    train_loader, test_loader = build_dataloader(args)
    model = MBNET(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 4. Create Trainer
    trainer = Trainer(model, optimizer, lr_scheduler, train_loader, args.epochs,
                      work_dir=args.work_dir, log_period=args.log_interval)
    # trainer.register_hooks([
    #     EvalHook(1, lambda: test(model, test_loader)),
    #     # Refer to inference_hook.py
    #     InferenceHook(test_loader.dataset)
    # ])
    trainer.train()


if __name__ == "__main__":
    main()
