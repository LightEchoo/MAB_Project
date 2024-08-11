import numpy as np
import matplotlib.pyplot as plt

# 定义节点参数
nodes = {
    'Node 1': {'mean': 52, 'variance': 3},
    'Node 2': {'mean': 48, 'variance': 3},
    'Node 3': {'mean': 47, 'variance': 3},
    'Node 4': {'mean': 49, 'variance': 3},
    'Node 5': {'mean': 50, 'variance': 3},
    'Node 6': {'mean': 55, 'variance': 3},
    'Node 7': {'mean': 60, 'variance': 3},
    'Node 8': {'mean': 53, 'variance': 3},
    'Node 9': {'mean': 54, 'variance': 3},
    'Node 10': {'mean': 58, 'variance': 3}
}

# 参数
N = len(nodes)  # 节点数量
T = 1000  # 试验次数
gamma_values = [0.1, 0.2, 0.3]  # EXP3的参数，控制探索和利用的平衡
alpha = 1  # 计算延迟奖励时使用的参数
m = 0.5  # 平衡延迟奖励和负载奖励

# 提取节点的均值和方差
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# 生成所有节点的延迟数据
def generate_all_iid_latency(means, variances, T):
    all_latencies = []
    for mean, variance in zip(means, variances):
        latencies = np.random.normal(loc=mean, scale=np.sqrt(variance), size=T)
        all_latencies.append(latencies)
    return np.array(all_latencies)

def generate_all_iid_loads(means, variances, T):
    all_loads = []
    for mean, variance in zip(means, variances):
        loads = np.random.normal(loc=mean, scale=np.sqrt(variance), size=T)
        all_loads.append(loads)
    return np.array(all_loads)

def calculate_cumulative_regret(regrets):
    return np.cumsum(regrets)

def smooth(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def calculate_rewards_latency(action_latency, alpha):
    reward_latency = np.exp(-alpha * action_latency)
    return reward_latency

def calculate_rewards_load(action_load):
    reward_load = 1 / (1 + (action_load))
    return reward_load

def get_reward(action_latency, action_load, alpha):
    reward_latency = calculate_rewards_latency(action_latency, alpha)
    reward_load = calculate_rewards_load(action_load)
    reward = m * reward_latency + (1 - m) * reward_load
    return reward

# 计算 Top-K 节点的准确率
def calculate_top_k_accuracy(actions, rewards_list, K):
    top_k_picks = 0
    for t in range(T):
        # 获取该时间步的所有节点的奖励
        rewards = rewards_list[t]

        # 找到奖励最大的前 K 个节点
        optimal_nodes = np.argsort(rewards)[-K:]

        # 如果选择的动作在前K个节点中则计数
        if actions[t] in optimal_nodes:
            top_k_picks += 1

    # 计算Top-K准确率
    top_k_accuracy = top_k_picks / T
    return top_k_accuracy

# 生成所有节点的延迟和负载数据
all_latencies = generate_all_iid_latency(true_means, true_variances, T)
all_loads = generate_all_iid_loads(true_means, true_variances, T)

# 初始化权重
results = {}

# EXP3算法
for gamma in gamma_values:
    weights = np.ones(N)  # 重置权重
    regrets = []
    actions = []
    rewards_list = []  # 用于保存每步的所有节点奖励

    for t in range(T):
        # 计算选择每个动作的概率
        probabilities = (1 - gamma) * (weights / np.sum(weights)) + (gamma / N)

        # 根据概率选择动作
        action = np.random.choice(np.arange(N), p=probabilities)

        # 获取动作对应的延迟和负载
        action_latency = all_latencies[action][t]
        action_load = all_loads[action][t]

        # 计算奖励
        reward = get_reward(action_latency, action_load, alpha)

        # 估计奖励
        estimated_reward = reward / probabilities[action]

        # 更新权重
        weights[action] *= np.exp(gamma * estimated_reward / N)

        # 计算每个节点的奖励
        rewards = []
        for node in range(N):
            node_latency = all_latencies[node][t]
            node_load = all_loads[node][t]
            node_reward = get_reward(node_latency, node_load, alpha)
            rewards.append(node_reward)

        rewards_list.append(rewards)

        # 选择奖励最大的节点
        optimal_reward = max(rewards)

        # 计算遗憾
        regret = optimal_reward - reward

        regrets.append(regret)
        actions.append(action)

    cumulative_regret = calculate_cumulative_regret(regrets)

    # 计算 Top-K 准确率
    top1_accuracy = calculate_top_k_accuracy(actions, rewards_list, 1)
    top2_accuracy = calculate_top_k_accuracy(actions, rewards_list, 2)
    top5_accuracy = calculate_top_k_accuracy(actions, rewards_list, 5)

    key = f"gamma={gamma}"
    results[key] = {
        'actions': actions,
        'regret_list': cumulative_regret,
        'regrets': regrets,
        'top1_accuracy': top1_accuracy,
        'top2_accuracy': top2_accuracy,
        'top5_accuracy': top5_accuracy,
    }

# 打印结果
for gamma in gamma_values:
    result = results[f'gamma={gamma}']
    print(f"Gamma: {gamma}")
    print(f"Top 1 Accuracy: {result['top1_accuracy']}")
    print(f"Top 2 Accuracy: {result['top2_accuracy']}")
    print(f"Top 5 Accuracy: {result['top5_accuracy']}")
    print()

# 绘制每个gamma的节点选择图像
for gamma in gamma_values:
    result = results[f'gamma={gamma}']
    plt.figure(figsize=(12, 8))
    plt.plot(result['actions'], label=f"Gamma: {gamma}")
    plt.xlabel('Steps')
    plt.ylabel('Node Selection')
    plt.legend()
    plt.title(f'Node Selection for Gamma = {gamma}')
    plt.show()

# 绘制每个gamma的单步遗憾图像
for gamma in gamma_values:
    result = results[f'gamma={gamma}']
    plt.figure(figsize=(12, 8))
    plt.plot(result['regrets'], label=f"Gamma: {gamma}")
    plt.xlabel('Steps')
    plt.ylabel('Single Step Regret')
    plt.legend()
    plt.title(f'Single Step Regret for Gamma = {gamma}')
    plt.show()

# 绘制包含所有gamma的累积遗憾图像
plt.figure(figsize=(12, 8))
for gamma in gamma_values:
    result = results[f'gamma={gamma}']
    plt.plot(result['regret_list'], label=f"Gamma: {gamma}")
plt.xlabel('Steps')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.title('Cumulative Regret for Different Gammas')
plt.show()

# 绘制包含所有gamma的Top-K节点选择准确性图像
K = 5  # 可以调整K值
ind = np.arange(len(gamma_values))  # x轴
width = 0.1  # 柱的宽度

fig, ax = plt.subplots(figsize=(12, 8))
top1_accuracies = [results[f'gamma={gamma}']['top1_accuracy'] for gamma in gamma_values]
top2_accuracies = [results[f'gamma={gamma}']['top2_accuracy'] for gamma in gamma_values]
top5_accuracies = [results[f'gamma={gamma}']['top5_accuracy'] for gamma in gamma_values]

top1_bar = ax.bar(ind - width, top1_accuracies, width, label='Top 1')
top2_bar = ax.bar(ind, top2_accuracies, width, label='Top 2')
top5_bar = ax.bar(ind + width, top5_accuracies, width, label='Top 5')

ax.set_xlabel('Gamma')
ax.set_ylabel('Accuracy')
ax.set_title('Top-K Node Selection Accuracy for Different Gammas')
ax.set_xticks(ind)
ax.set_xticklabels([f'{gamma}' for gamma in gamma_values])
ax.legend()

plt.show()

# 绘制包含所有gamma的平滑单步遗憾图像
plt.figure(figsize=(12, 8))
for gamma in gamma_values:
    result = results[f'gamma={gamma}']
    smoothed_regrets = smooth(result['regrets'])
    plt.plot(smoothed_regrets, label=f"Gamma: {gamma}")
plt.xlabel('Time Steps')
plt.ylabel('Smoothed Single-Step Regret')
plt.legend()
plt.title('EXP3 Algorithm: Smoothed Single-Step Regret Over Time')
plt.show()
