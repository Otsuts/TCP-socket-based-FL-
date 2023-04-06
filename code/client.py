import torch
import dill
import os
import torch.nn as nn
import socket
import random
import pickle
import argparse
import numpy as np
from torch.optim import Adam
from model import CNN


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Client():
    def __init__(self, number, path='../'):
        self.model = CNN()
        self.number = number
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.data_path = os.path.join(path, 'data', f'Client{number}.pkl')
        self.model_path = os.path.join(path, 'models', f'Client{number}.pth')
        with open(self.data_path, 'rb') as f:
            self.train_dataset = dill.load(f)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def receive_model(self):
        self.model.load_state_dict(torch.load(os.path.join('../models', 'handout_model.pth')))

    def send_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def train_one_epoch(self):
        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()


class Client_TCP():
    def __init__(self, number, path='../'):
        self.model = CNN()
        self.number = number
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.data_path = os.path.join(path, 'data', f'Client{number}.pkl')
        self.model_path = os.path.join(path, 'models', f'Client{number}.pth')
        with open(self.data_path, 'rb') as f:
            self.train_dataset = dill.load(f)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=100, shuffle=True)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        # 建立套接字
        self.client = socket.socket()
        self.address = ('127.0.0.1', 8085)
        self.client.connect(self.address)
        # 获取初始下发的模型参数
        model_param = pickle.loads(self.receive_long_data())
        # 用下发的模型参数训练一轮
        self.train_one_epoch(model_param)
        # 把模型参数发回去
        self.send_param()

    def receive_long_data(self):
        '''
        处理过长的tcp内容
        :return:
        '''
        total_data = bytes()
        while True:
            data = self.client.recv(1024)
            total_data += data
            if len(data) < 1024:
                break
        return total_data

    def train_one_epoch(self, model_param):
        self.model.load_state_dict(model_param)
        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

    def send_param(self):
        # 把训练好的模型参数一股脑发回去
        self.client.sendall(pickle.dumps(self.model.state_dict()))


if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_idx', type=int)
    args = parser.parse_args()
    client = Client_TCP(args.client_idx)
