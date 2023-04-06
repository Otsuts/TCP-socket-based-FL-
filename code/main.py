import argparse
import random
import os
import numpy as np
import torch.utils.data
from client import Client
from server import Server_PP, Server_TCP
import torchvision


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--num_part', type=int, default=15)
    parser.add_argument('--method', type=str, default='no')

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FedTrainer():
    def __init__(self, args):
        self.args = args
        self.clients = [Client(x + 1) for x in range(args.num_clients)]
        self.server = Server_PP(args.num_clients)
        self.test_dataset = torchvision.datasets.MNIST(
            '../data', train=False, download=True, transform=torchvision.transforms.ToTensor()
        )
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=256, shuffle=False)

    def evaluate(self):
        correct = 0
        for data, target in self.test_loader:
            with torch.no_grad():
                pred = self.server.model(data)
                prediction = torch.argmax(pred, dim=1)
                correct += sum(prediction == target)

        return correct / self.test_dataset.__len__()

    def train(self):
        for epoch in range(self.args.epoch):
            update_index = random.sample(range(self.args.num_clients), k=args.num_part)
            self.server.send_model(update_index)
            for client_index in update_index:
                self.clients[client_index].receive_model()
                self.clients[client_index].train_one_epoch()
                self.clients[client_index].send_model()

            self.server.receive_model(update_index)
            self.server.update(update_index)
            precision = self.evaluate()
            print(f'Epoch: {epoch} ,precision: {precision}')


class FedTrainer_TCP():
    def __init__(self):
        self.server = Server_TCP(args.num_clients)
        self.test_dataset = torchvision.datasets.MNIST(
            '../data', train=False, download=True, transform=torchvision.transforms.ToTensor()
        )
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=256, shuffle=False)

    def evaluate(self):
        correct = 0
        for data, target in self.test_loader:
            with torch.no_grad():
                pred = self.server.model(data)
                prediction = torch.argmax(pred, dim=1)
                correct += sum(prediction == target)

        return correct / self.test_dataset.__len__()

    def train(self):
        for epoch in range(20):
            self.train_one_epoch()
            precision = self.evaluate()
            print(f'Epoch: {epoch} ,precision: {precision}')

    def train_one_epoch(self):
        self.server.accept_client()
        self.server.update()


def main(args):
    set_seed()
    if args.method == 'TCP':
        train = FedTrainer_TCP()
    else:
        train = FedTrainer(args)
    train.train()


if __name__ == '__main__':
    args = get_args()
    main(args)
