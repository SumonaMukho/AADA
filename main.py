import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
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
parser.add_argument("--batch_size", default=128, type=int, help="Size of batch")
parser.add_argument("--learning_rate", default=0.0002, type=float)
parser.add_argument("--max_round", default=30, type=int)
parser.add_argument("--nb_experiments", default=1, type=int)
parser.add_argument("--b", default=10, type=int, help="budget per round")
parser.add_argument("--alpha", default=0.1, type=float, help="Parameter of the mixture of losses")
parser.add_argument('--cuda', dest='cuda', action='store_true')
parser.add_argument('--no-cuda', dest='cuda', action='store_false')
parser.set_defaults(cuda=torch.cuda.is_available())
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--ent_reg', dest='ent_reg', action='store_true')
parser.set_defaults(ent_reg=False)
parser.add_argument('--reg_lambda', default=0.001, type=float)

args = parser.parse_args()

torch.random.manual_seed(42)

net = CNNModel()
if args.cuda:
    net.cuda()
    print("Using CUDA")
    cudnn.benchmark = True
else:
    print("Using CPU")
    
for p in net.parameters():
    p.requires_grad = True

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
        print("\nRound",round+1)
        print("Picking labels : ")
        idxs, scores = score(net, args, target_train_loader, known_labels)
        new_labels = scores.argsort()[-args.b:]
        new_labels = idxs[new_labels]
        known_labels += new_labels.tolist()
        assert len(known_labels)==(round+1)*args.b
        print("\tPicked",len(new_labels),"labels")

        # Retrain
        print("Training on new labels")
        net = train(net, args, optimizer, source_train_loader, target_train_loader, known_labels)
        print("Testing :")
        test(net,args,target_test_loader)
    print("Test for results of experiment ", experiment)
