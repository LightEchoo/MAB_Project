from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataGenerateConfig:
    mean_load: float = 50.0  # Mean load of nodes' load
    var_load: float = 10.0  # Variance load of nodes' load
    iid_var: float = 1.0  # Variance for iid data
    theta: float = 0.9  # AR(1) parameter


@dataclass
class ARConfig:
    order: int = 5  # AR order


@dataclass
class LSTMConfig:
    hidden_size: int = 64  # Hidden size
    num_layers: int = 4  # Number of layers


@dataclass
class GATConfig:
    hidden_size: int = 32  # Hidden size
    num_heads: int = 8  # Number of attention heads
    num_gat_layers: int = 3  # Number of GAT layers


@dataclass
class GNNConfig:
    hidden_size: int = 32
    num_layers: int = 3


@dataclass
class Config:
    N: int = 10  # Number of nodes
    T: int = 11000  # Total time steps

    T_train_val: int = 10000  # Training and validation time steps
    train_ratio: float = 0.8  # Training ratio
    T_train: int = 8000  # Training time steps
    T_val: int = 2000  # Validation time steps

    T_test: int = 1000  # Test time steps
    data_type: str = 'ar1'  # 'iid' or 'ar1'

    batch_size: int = 64  # Batch size
    seq_length: int = 20  # Sequence length
    input_size: int = 10  # Input size
    output_size: int = 10  # Output size
    learning_rate: float = 0.001  # Learning rate
    num_epochs: int = 100  # Number of epochs
    num_workers: int = 24  # Number of workers for DataLoader
    device: str = 'cuda'  # Device
    mix_precision: bool = True  # Mixed precision training

    # 早停的参数
    patience_epochs: int = 6  # 'patience_epochs' 个 epoch 没有提升，就停止训练
    min_delta: float = 1e-2  # 当监控指标的变化小于 min_delta 时，就视为没有提升

    # 调度器的参数
    mode: str = 'min'  # 'min' 表示监控指标的值越小越好，'max' 表示监控指标的值越大越好
    factor: float = 0.1  # 学习率调度器的缩放因子
    patience_lr: int = 2  # 'patience_lr' 个 epoch 没有提升，就缩放学习率
    min_lr: float = 1e-6  # 学习率的下限
    threshold: float = 1e-2  # 监控指标的变化小于 threshold 时，就视为没有提升

    # 使用 default_factory 来实例化复杂类型
    dg_config: DataGenerateConfig = field(default_factory=DataGenerateConfig)
    ar_config: ARConfig = field(default_factory=ARConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    gat_config: GATConfig = field(default_factory=GATConfig)
    gnn_config: GNNConfig = field(default_factory=GNNConfig)

    def print_config_info(self):
        print('-----------------Config Info-----------------')
        self._recursive_print(vars(self))

    def _recursive_print(self, config_dict, indent=0):
        for key, value in config_dict.items():
            if isinstance(value, (DataGenerateConfig, ARConfig, LSTMConfig, GATConfig)):
                print(" " * indent + f"{key}:")
                self._recursive_print(vars(value), indent + 4)
            else:
                print(" " * indent + f"{key}: {value}")


class DataGenerate:
    def __init__(self, config: Config):
        self.config = config  # 配置
        self.means_loads = self._generate_means()  # 生成节点的平均负载

        self.load_iid, self.mean_iid = self._generate_iid_data()  # 生成iid数据
        self.load_ar1, self.mean_ar1 = self._generate_ar1_data()  # 生成ar1数据

        self._save_data()  # 保存数据

        self.print_data_generate_info()  # 打印信息
        self.plot_original_means()  # 绘制原始平均负载

    def plot_original_means(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.means_loads, marker='o', linestyle='-', color='b', label='means_load')
        plt.title('Original Random Means of Nodes of Load')
        plt.xlabel('Node')
        plt.ylabel('Mean Load')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _generate_means(self):
        return np.random.normal(self.config.dg_config.mean_load, self.config.dg_config.var_load, size=(self.config.N,))

    def _generate_iid_data(self):
        loads = np.array([np.random.normal(loc=self.means_loads[i], scale=self.config.dg_config.iid_var,
                                           size=self.config.T_train_val + self.config.T_test) for i in
                          range(self.config.N)])
        return loads, np.mean(loads, axis=1)

    def _generate_ar1_data(self):
        loads = np.zeros((self.config.N, self.config.T_train_val + self.config.T_test))

        def generate_ar1(mean_node):
            ar1 = torch.zeros(self.config.T_train_val + self.config.T_test)
            ar1[0] = mean_node
            for t in range(1, self.config.T_train_val + self.config.T_test):
                ar1[t] = self.config.dg_config.theta * ar1[t - 1] + (
                        1 - self.config.dg_config.theta) * mean_node + np.random.normal(0, 1)
            return ar1

        for i in range(self.config.N):
            loads[i] = generate_ar1(self.means_loads[i])

        return loads, np.mean(loads, axis=1)

    def _save_data(self):
        pd.DataFrame(self.load_iid).to_csv('load_iid_data.csv', index=False)
        pd.DataFrame(self.load_ar1).to_csv('load_ar1_data.csv', index=False)

    def print_data_generate_info(self):
        print('-----------------Data Generate Info-----------------')
        print('means_loads.shape:', self.means_loads.shape)
        print('load_iid.shape:', self.load_iid.shape)
        print('mean_iid.shape:', self.mean_iid.shape)
        print('load_ar1.shape:', self.load_ar1.shape)
        print('mean_ar1.shape:', self.mean_ar1.shape)


class TrainVaildManage:
    def __init__(self, config: Config):
        self.config = config
        self.load_iid = pd.read_csv('load_iid_data.csv').values
        self.load_ar1 = pd.read_csv('load_ar1_data.csv').values

        if self.config.data_type == 'iid':
            self.data_np = self.load_iid
        elif self.config.data_type == 'ar1':
            self.data_np = self.load_ar1

        self.data_tensor = torch.tensor(self.data_np, device=self.config.device, dtype=torch.float32)

        # 从config中获取数据的时间步信息
        self.T = self.config.T
        self.T_train = self.config.T_train
        self.T_val = self.config.T_val
        self.train_ratio = self.config.train_ratio
        self.T_train_val = self.config.T_train_val
        self.T_test = self.config.T_test

        # 划分np.array的训练集、验证集和测试集
        self.train_val_data_np = self.data_np[:, :self.T_train_val]
        self.train_data_np = self.data_np[:, :self.T_train]
        self.val_data_np = self.data_np[:, self.T_train:self.T_train_val]
        self.test_data_np = self.data_np[:, self.T_train_val:]

        # 储存tensor的训练集、验证集和测试集
        self.train_val_data_tensor = torch.tensor(self.train_val_data_np, device=self.config.device, dtype=torch.float32)
        self.train_data_tensor = torch.tensor(self.train_data_np, device=self.config.device, dtype=torch.float32)
        self.val_data_tensor = torch.tensor(self.val_data_np, device=self.config.device, dtype=torch.float32)
        self.test_data_tensor = torch.tensor(self.test_data_np, device=self.config.device, dtype=torch.float32)

        # 创建训练集和验证集的序列数据，用于训练和验证
        self.train_val_x, self.train_val_y = self._create_sequences(self.train_val_data_np)
        self.train_x, self.train_y = self._create_sequences(self.train_data_np)
        self.val_x, self.val_y = self._create_sequences(self.val_data_np)

        # 创建TensorDataset，用于创建DataLoader
        self.train_val_dataset = TensorDataset(self.train_val_x, self.train_val_y)
        self.train_dataset = TensorDataset(self.train_x, self.train_y)
        self.val_dataset = TensorDataset(self.val_x, self.val_y)

        # 创建数据加载器
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        self.train_val_dataloader = DataLoader(self.train_val_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)

        # 打印信息
        self.print_train_valid_info()
        self.print_dataloader_info(self.train_dataloader, title='Train')
        self.print_dataloader_info(self.val_dataloader, title='Valid')
        self.print_dataloader_info(self.train_val_dataloader, title='Train and Valid')

        # 创建GNN的边索引
        self.edge_index_tensor = torch.tensor(
            np.array([(i, j) for i in range(self.config.N) for j in range(self.config.N)]).T,
            dtype=torch.long)  # 默认全连接图

    def _create_sequences(self, data):
        x, y = [], []

        # 循环次数不是T_train - seq_length + 1，因为训练集里并没有第10001个真实数据作为target。
        # 最后一次生成的序列会在逐步更新的过程中使用，而不是在初始训练集中。
        for i in range(data.shape[1] - self.config.seq_length):
            x.append(data[:, i: i + self.config.seq_length].T)
            y.append(data[:, i + self.config.seq_length])
        return torch.tensor(np.array(x)), torch.tensor(np.array(y))

    def print_train_valid_info(self):
        print('-----------------Train and Valid Data Info-----------------')
        print('load_iid.shape:', self.load_iid.shape)
        print('load_ar1.shape:', self.load_ar1.shape)
        print('data_type:', self.config.data_type)
        print('data_np.shape:', self.data_np.shape)
        print('data_tensor.shape:', self.data_tensor.shape)

        print('T:', self.T)
        print('T_train:', self.T_train)
        print('T_val:', self.T_val)
        print('train_ratio:', self.train_ratio)
        print('T_train_val:', self.T_train_val)
        print('T_test:', self.T_test)

        print('train_val_data_np.shape:', self.train_val_data_np.shape)
        print('train_data_np.shape:', self.train_data_np.shape)
        print('val_data_np.shape:', self.val_data_np.shape)
        print('test_data_np.shape:', self.test_data_np.shape)

        print('train_val_data_tensor.shape:', self.train_val_data_tensor.shape)
        print('train_data_tensor.shape:', self.train_data_tensor.shape)
        print('val_data_tensor.shape:', self.val_data_tensor.shape)
        print('test_data_tensor.shape:', self.test_data_tensor.shape)

        print('train_val_x.shape:', self.train_val_x.shape)
        print('train_val_y.shape:', self.train_val_y.shape)
        print('train_x.shape:', self.train_x.shape)
        print('train_y.shape:', self.train_y.shape)
        print('val_x.shape:', self.val_x.shape)
        print('val_y.shape:', self.val_y.shape)


    def print_dataloader_info(self, dataloader, title='Dataloader Info'):
        print(f'-------------{title} Dataloader Info-------------')
        for i, (x, y) in enumerate(dataloader):
            if i % 30 == 0 or i == len(dataloader) - 1:
                print(f'{title} i: {i:>3}, x: {x.shape}, y: {y.shape}')

    def plot_range_data(self, data, start=None, end=None, title='Load Data'):
        start = 0 if start is None else start  # 默认从0开始
        end = data.shape[1] if end is None else end  # 默认到最后结束

        # 绘制指定范围内的数据
        time_steps = np.arange(start, end)

        plt.figure(figsize=(12, 6))
        for i in range(data.shape[0]):
            plt.plot(time_steps, data[i, start:end], label=f'Node {i}')
        plt.title(f'{title} - Nodes {0}-{data.shape[0]}')
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def Process(RE_GENERATE_DATA=False, **kwargs):
    RE_GENERATE_DATA = RE_GENERATE_DATA  # 是否重新生成数据

    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # 默认配置
    default_config = {
        'N': 10,
        'T': 11000,
        'T_train_val': 10000,
        'T_test': 1000,
        'train_ratio': 0.8,
        'T_train': 8000,
        'T_val': 2000,
        'data_type': 'ar1',

        'batch_size': 64,
        'seq_length': 20,
        'input_size': 10,
        'output_size': 10,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'num_workers': 24,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'mix_precision': True if device == 'cuda' else False,

        'patience_epochs': 6,
        'min_delta': 1e-2,

        'mode': 'min',
        'factor': 0.1,
        'patience_lr': 2,
        'min_lr': 1e-6,
        'threshold': 1e-2,

        'dg_config': DataGenerateConfig(mean_load=50.0, var_load=10.0, iid_var=1.0, theta=0.9),
        'ar_config': ARConfig(order=5),
        'lstm_config': LSTMConfig(hidden_size=64, num_layers=4),
        'gat_config': GATConfig(hidden_size=32, num_heads=8, num_gat_layers=3),
        'gnn_config': GNNConfig(hidden_size=32, num_layers=3)
    }

    # 更新默认配置
    default_config.update(kwargs)

    # 使用更新后的配置创建Config对象
    config = Config(**default_config)

    config.print_config_info()

    if RE_GENERATE_DATA:
        data_generate = DataGenerate(config)

    data_train_val_manage = TrainVaildManage(config)

    data_train_val_manage.plot_range_data(data_train_val_manage.data_np[:3, :], title='Data')
    # data_train_val_manage.plot_range_data(data_train_val_manage.train_data_np[:3, :], title='Train Data')
    # data_train_val_manage.plot_range_data(data_train_val_manage.val_data_np[:3, :], title='Valid Data')
    # data_train_val_manage.plot_range_data(data_train_val_manage.test_data_np[:3, :], title='Test Data')

    # data_manage = DataManage(config)
    #
    # data_manage.plot_range_data(data_manage.train_val_data, 0, 1000, title='Train and Valid Data')
    # data_manage.plot_range_data(data_manage.test_data, 0, 1000, title='Test Data')
    if RE_GENERATE_DATA:
        return config, data_generate, data_train_val_manage
    else:
        return config, data_train_val_manage


if __name__ == '__main__':
    current_config = {
        'N': 10,
        'T': 11000,
        'T_train_val': 10000,
        'T_test': 1000,
        'train_ratio': 0.8,
        'T_train': 8000,
        'T_val': 2000,
        'data_type': 'ar1',

        'batch_size': 64,
        'seq_length': 20,
        'input_size': 10,
        'output_size': 10,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'num_workers': 24,
        'device': 'cuda',

        'patience_epochs': 6,
        'min_delta': 1e-2,

        'mode': 'min',
        'factor': 0.1,
        'patience_lr': 2,
        'min_lr': 1e-6,
        'threshold': 1e-2,

        'dg_config': DataGenerateConfig(mean_load=50.0, var_load=10.0, iid_var=1.0, theta=0.9),
        'ar_config': ARConfig(order=5),
        'lstm_config': LSTMConfig(hidden_size=64, num_layers=4),
        'gat_config': GATConfig(hidden_size=32, num_heads=8, num_gat_layers=3),
        'gnn_config': GNNConfig(hidden_size=32, num_layers=3)
    }
    # config, data_generate, data_train_val_manage = main(RE_GENERATE_DATA=True, **current_config)

    config, data_manage = Process(RE_GENERATE_DATA=False, **current_config)
#%%
