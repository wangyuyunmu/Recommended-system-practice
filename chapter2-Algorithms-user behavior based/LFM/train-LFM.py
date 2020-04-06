# 导入包
import random
import math
import numpy as np
import time
from tqdm import tqdm, trange

from Recommend.ml_1m.chapter2.LFM.metrics import Metric
from Recommend.ml_1m.chapter2.LFM.Dataset import Dataset

# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper



def LFM(train, ratio, K, lr, step, lmbda, N):
    '''
    :params: train, 训练数据
    :params: ratio, 负采样的正负比例
    :params: K, 隐语义个数
    :params: lr, 初始学习率
    :params: step, 迭代次数
    :params: lmbda, 正则化系数
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''

    all_items = {}
    for user in train:
        for item in train[user]:
            if item not in all_items:
                all_items[item] = 0
            all_items[item] += 1
    all_items = list(all_items.items())
    items = [x[0] for x in all_items]
    pops = [x[1] for x in all_items]



    # 负采样函数(注意！！！要按照流行度进行采样)
    def nSample(data, ratio):
        new_data = {}
        # 正样本
        for user in data:
            if user not in new_data:
                new_data[user] = {}
            for item in data[user]:
                new_data[user][item] = 1

        # # 分步按照流行度采集负样本
        # for user in new_data:
        #     seen = set(new_data[user])
        #     pos_num = len(seen)
        #     n = 0
        #     for i in range(pos_num*3):
        #         # temp = items[np.random.randint(0,len(items)-1)]
        #         temp = np.random.choice(items, 1, pops)[0]
        #         if temp in new_data[user]:
        #             continue
        #         new_data[user][temp] = 0
        #         n += 1
        #         if n > pos_num*ratio:
        #             break

        # 负样本
        for user in new_data:
            seen = set(new_data[user])
            pos_num = len(seen)
            item = np.random.choice(items, int(pos_num * ratio * 3), pops)
            item = [x for x in item if x not in seen][:int(pos_num * ratio)]
            new_data[user].update({x: 0 for x in item})


        return new_data

    # 训练
    P, Q = {}, {}
    for user in train:
        P[user] = np.random.random(K)
    for item in items:
        Q[item] = np.random.random(K)

    for s in trange(step):
        data = nSample(train, ratio)
        for user in data:
            for item in data[user]:
                eui = data[user][item] - (P[user] * Q[item]).sum()
                P[user] += lr * (Q[item] * eui - lmbda * P[user])
                Q[item] += lr * (P[user] * eui - lmbda * Q[item])
        lr *= 0.9  # 调整学习率

    # 获取接口函数
    def GetRecommendation(user):
        seen_items = set(train[user])
        recs = {}
        for item in items:
            if item not in seen_items:
                recs[item] = (P[user] * Q[item]).sum()
        recs = list(sorted(recs.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, ratio=1,
                 K=100, lr=0.02, step=50, lmbda=0.01, fp='../../ratings.dat'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: ratio, 正负样本比例
        :params: K, 隐语义个数
        :params: lr, 学习率
        :params: step, 训练步数
        :params: lmbda, 正则化系数
        :params: fp, 数据文件路径
        '''
        self.M = M
        self.K = K
        self.N = N
        self.ratio = ratio
        self.lr = lr
        self.step = step
        self.lmbda = lmbda
        self.fp = fp
        self.alg = LFM

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg(train, self.ratio, self.K,
                                     self.lr, self.step, self.lmbda, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, ratio={}): {}'.format( \
            self.M, self.N, self.ratio, metrics))

# LFM实验(运行时间较长，这里没贴实验结果)
M, N = 8, 10
for r in [5, 10, 20]:
    exp = Experiment(M, N, ratio=r)
    exp.run()