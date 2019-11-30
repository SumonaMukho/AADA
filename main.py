import argparse
import torch
import torch.optim as optim
from load_data import *
from models.model import CNNModel
from models.functions import train, test, score
import numpy as np

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", default=10, type=int, help="Number of classes")
parser.add_argument("--source_dataset", default="svhn", type=str, help="Name of the source dataset")
parser.add_argument("--target_dataset", default="mnist", type=str, help="Name of the target dataset")
parser.add_argument("--n_epochs", default=60, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=100, type=int, help="Size of batch")
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--max_round", default=5, type=int)
parser.add_argument("--nb_experiments", default=1, type=int)
parser.add_argument("--b", default=10, type=int, help="budget per round")
parser.add_argument("--cuda", default=torch.cuda.is_available(), type=bool, help="Use of CUDA")

args = parser.parse_args()

torch.random.manual_seed(42)

net = CNNModel()
if args.cuda:
    net.cuda()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

# Pass to cuda

known_labels = []

# Download dataset
source_train = iSVHN(root="./data/", train=True, transform=svhn_transform, download=True)
source_test = iSVHN(root="./data/", train=False, transform=svhn_transform, download=True)
target_train = iMNIST(root="./data/", train=True, transform=mnist_transform, download=True)
target_test = iMNIST(root="./data/", train=False, transform=mnist_transform, download=True)
source_train_loader = torch.utils.data.DataLoader(source_train, batch_size=args.batch_size, num_workers=2)
source_test_loader = torch.utils.data.DataLoader(source_test, batch_size=args.batch_size, num_workers=2)
target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=args.batch_size, num_workers=2)
target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=args.batch_size, num_workers=2)


# The process
for experiment in range(args.nb_experiments):
    print("First training")
    net = train(net, args, optimizer, source_train_loader, target_train_loader, None)


    # Active Learning process
    for round in range(args.max_round):
        print("Picking labels")
        scores = score(net, args, target_train_loader.dataset.get_data(), target_train_loader.dataset.targets)
        scores = scores.cpu().numpy()
        new_labels = scores.argsort()[-args.b:]
        known_labels.append(new_labels)
        print(len(known_labels),(round+1)*args.b)
        assert len(known_labels)==(round+1)*args.b
        print("\tPicked ",new_labels)

        # Retrain
        print("Training on new labels")
        net = train(net, args, optimizer, source_train_loader, target_train_loader, known_labels)
        print("Validating")
        test(net,args,target_test_loader)
    print("Test for results of experiment ", experiment)
