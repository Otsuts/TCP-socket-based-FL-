import collections
import socket
import time
from threading import Thread
import pickle
import torch
import os
from model import CNN


class Server():
    def __init__(self, client_number):
        self.model = CNN()
        self.client_number = client_number
        self.sub_models = [CNN() for _ in range(client_number)]

    def send_model(self, update_index):
        torch.save(self.model.state_dict(), os.path.join('../models', 'handout_model.pth'))

    def receive_model(self, update_index):
        for index, model in enumerate(self.sub_models):
            model.load_state_dict(torch.load(os.path.join('../models', f'Client{index + 1}.pth')))

    def update(self, update_index):
        weight_keys = list(self.sub_models[0].state_dict().keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for model in self.sub_models:
                key_sum += model.state_dict()[key]

            fed_state_dict[key] = key_sum / self.client_number

        self.model.load_state_dict(fed_state_dict)


class Server_PP(Server):
    def __init__(self, client_number):
        super().__init__(client_number)

    def receive_model(self, update_index):
        for index in update_index:
            self.sub_models[index].load_state_dict(torch.load(os.path.join('../models', f'Client{index + 1}.pth')))

    def update(self, update_index):
        weight_keys = list(self.sub_models[0].state_dict().keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for model_index in update_index:
                key_sum += self.sub_models[model_index].state_dict()[key]

            fed_state_dict[key] = key_sum / self.client_number

        self.model.load_state_dict(fed_state_dict)


class Server_TCP():
    def __init__(self, client_number):
        self.client_number = client_number
        self.model = CNN()
        # TCP 通讯相关
        self.address = ('127.0.0.1', 8085)
        self.socket_server = socket.socket()
        self.socket_server.bind(self.address)
        self.socket_server.listen(5)
        # 接受client的参数字典
        self.fed_state_dict = collections.OrderedDict()

    def accept_client(self):
        # 在一轮训练中与各个client通信并交换参数
        # 计数：有多少客户端已经成功连接了
        updated_model = 0
        while True:
            # 等待客户端连接
            client, info = self.socket_server.accept()
            updated_model += 1
            time.sleep(0.5)
            # 给每个客户端创建一个独立的线程进行管理
            thread = Thread(target=self.accept_and_update, args=(client, info))
            # 设置成守护线程，在主进程退出时可以直接退出
            thread.setDaemon(True)
            thread.start()
            # 所有client都通信完毕，跳出循环
            if updated_model == self.client_number:
                break

    def accept_and_update(self, client, info):
        '''
        子线程里面要干的事情
        '''
        # 把初始化模型下发到子线程
        client.sendall(pickle.dumps(self.model.state_dict()))
        while True:
            # 接收子线程训练好的模型
            received_params = pickle.loads(self.receive_long_data(client))
            # 把新接受的参数添加到server的参数里面
            self.update_one_model(received_params)
            # 已经接受到模型参数：这个子线程可以废了
            if received_params:
                break

    def receive_long_data(self, client):
        '''
        TCP一个数据报只能接收1024字节，所以需要把多次接受的拼起来，返回整个数据
        '''
        total_data = bytes()
        while True:
            data = client.recv(1024)
            total_data += data
            if len(data) < 1024:
                break
        return total_data

    def update_one_model(self, param):
        '''
        根据单个client训练好的模型进行参数更新
        :param param: dict，client的模型参数
        :return: None
        '''
        weight_keys = list(self.model.state_dict().keys())
        for key in weight_keys:
            if key in self.fed_state_dict.keys():
                self.fed_state_dict[key] += param[key]
            else:
                self.fed_state_dict[key] = param[key]

    def update(self):
        # avg pooling
        for value in self.fed_state_dict.values():
            value /= self.client_number
        self.model.load_state_dict(self.fed_state_dict)
        self.fed_state_dict.clear()


if __name__ == '__main__':
    server = Server(20)
    server.update(range(20))
    assert 0
