from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
    hidden_size: int = 50  # Hidden size
    num_layers: int = 4  # Number of layers


@dataclass
class GNNConfig:
    # Add relevant fields if necessary
    pass


@dataclass
class Config:
    N: int = 10  # Number of nodes
    T_train_val: int = 10000  # Training and validation time steps
    T_test: int = 1000  # Test time steps
    data_type: str = 'ar1'  # 'iid' or 'ar1'

    batch_size: int = 64  # Batch size
    seq_length: int = 20  # Sequence length
    input_size: int = 10  # Input size
    output_size: int = 10  # Output size
    learning_rate: float = 0.001  # Learning rate
    num_epochs: int = 100  # Number of epochs
    num_workers: int = 24  # Number of workers for DataLoader
    mix_precision: bool = True  # Mixed precision training
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device

    # 使用 default_factory 来实例化复杂类型
    dg_config: DataGenerateConfig = field(default_factory=DataGenerateConfig)
    ar_config: ARConfig = field(default_factory=ARConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    gnn_config: GNNConfig = field(default_factory=GNNConfig)

    def print_config_info(self):
        print("Config settings:")
        self._recursive_print(vars(self))

    def _recursive_print(self, config_dict, indent=0):
        for key, value in config_dict.items():
            if isinstance(value, (DataGenerateConfig, ARConfig, LSTMConfig, GNNConfig)):
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
        print('means_loads:', self.means_loads.shape)
        print('load_iid:', self.load_iid.shape)
        print('mean_iid:', self.mean_iid.shape)
        print('load_ar1:', self.load_ar1.shape)
        print('mean_ar1:', self.mean_ar1.shape)


class DataManage:
    def __init__(self, config: Config):
        self.config = config  # 配置
        self.load_iid = pd.read_csv('load_iid_data.csv').values  # 加载iid数据, shape: (N, T), 10*11000
        self.load_ar1 = pd.read_csv('load_ar1_data.csv').values  # 加载ar1数据, shape: (N, T), 10*11000

        if self.config.data_type == 'iid':
            self.data = self.load_iid
        elif self.config.data_type == 'ar1':
            self.data = self.load_ar1

        self.train_val_data = self.data[:, :self.config.T_train_val]  # 训练集和验证集数据, shape: (N, T_train_val), 10*10000
        self.test_data = self.data[:, self.config.T_train_val:]  # 测试集数据, shape: (N, T_test), 10*1000

        # 获取训练集和验证集的数据
        self.train_sets, self.val_sets = self._create_sequences()

        # 创建数据集
        self.train_val_dataset = TensorDataset(self.train_sets, self.val_sets)
        self.dataloader = DataLoader(self.train_val_dataset, batch_size=self.config.batch_size, shuffle=True,
                                     num_workers=self.config.num_workers)

        self.print_data_manage_info()  # 打印信息
        self.print_dataloader_info()  # 打印dataloader信息

    def _create_sequences(self):
        train_sets = []
        val_sets = []
        for i in range(self.config.T_train_val - self.config.seq_length):
            # 循环次数不是T_train - seq_length + 1，因为训练集里并没有第10001个真实数据作为target。
            # 最后一次生成的序列会在逐步更新的过程中使用，而不是在初始训练集中。
            train = self.train_val_data[:, i: i + self.config.seq_length].T  # 提取每个时间步的序列
            val = self.train_val_data[:, i + self.config.seq_length]  # 提取目标值
            train_sets.append(train)
            val_sets.append(val)
        return torch.tensor(np.array(train_sets)), torch.tensor(np.array(val_sets))

    def print_data_manage_info(self):
        print('load_iid:', self.load_iid.shape)
        print('load_ar1:', self.load_ar1.shape)
        print('data:', self.data.shape)
        print('train_val_data:', self.train_val_data.shape)
        print('test_data:', self.test_data.shape)
        print('train_sets:', self.train_sets.shape)
        print('val_sets:', self.val_sets.shape)
        print('train_val_dataset:', len(self.train_val_dataset))
        print('dataloader:', len(self.dataloader))

    def print_dataloader_info(self):
        for i, (train, val) in enumerate(self.dataloader):
            if i % 30 == 0 or i == len(self.dataloader) - 1:  # 每30次打印一次，确保最后一次打印
                print(f'i: {i:>3}, train: {train.shape}, val: {val.shape}')


if __name__ == '__main__':
    RE_GENERATE_DATA = False  # 是否重新生成数据

    # 默认配置
    config = Config(
        N=10,
        T_train_val=10000,
        T_test=1000,
        data_type='ar1',

        batch_size=64,
        seq_length=20,
        input_size=10,
        output_size=10,
        learning_rate=0.001,
        num_epochs=100,
        num_workers=24,
        mix_precision=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',

        dg_config = DataGenerateConfig(mean_load=50.0, var_load=10.0, iid_var=1.0, theta=0.9),
        ar_config = ARConfig(order=5),
        lstm_config = LSTMConfig(hidden_size=50, num_layers=4),
        gnn_config = GNNConfig(),
    )

    config.print_config_info()

    if RE_GENERATE_DATA:
        data_generate = DataGenerate(config)

    data_manage = DataManage(config)

#%%
