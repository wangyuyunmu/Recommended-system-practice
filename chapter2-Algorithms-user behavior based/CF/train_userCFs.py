import random
import math
import time
from tqdm import tqdm
from code_w.recommand.chapter2.Dataset import Dataset
from code_w.recommand.chapter2.metrics import Metric

# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper

# 算法实现 random、mostpopular、userCF、userIIF
# 1. 随机推荐
def Random(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    items = {}
    for user in train:
        for item in train[user]:
            items[item] = 1

    def GetRecommendation(user):
        # 随机推荐N个未见过的
        user_items = set(train[user])
        rec_items = {k: items[k] for k in items if k not in user_items}
        rec_items = list(rec_items.items())
        random.shuffle(rec_items)
        return rec_items[:N]

    return GetRecommendation


# 2. 热门推荐
def MostPopular(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    items = {}
    for user in train:
        for item in train[user]:
            if item not in items:
                items[item] = 0
            items[item] += 1

    def GetRecommendation(user):
        # 随机推荐N个没见过的最热门的
        user_items = set(train[user])
        rec_items = {k: items[k] for k in items if k not in user_items}
        rec_items = list(sorted(rec_items.items(), key=lambda x: x[1], reverse=True))
        return rec_items[:N]

    return GetRecommendation


# 3. 基于用户余弦相似度的推荐
def UserCF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似用户数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算item->user的倒排索引
    item_users = {}
    for user in train:
        for item in train[user]:
            if item not in item_users:
                item_users[item] = []
            item_users[item].append(user)

    # 计算用户相似度矩阵
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i: continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True))
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for u, _ in sorted_user_sim[user][:K]:
            for item in train[u]:
                # 要去掉用户见过的
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


# 4. 基于改进的用户余弦相似度的推荐
def UserIIF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似用户数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算item->user的倒排索引
    item_users = {}
    for user in train:
        for item in train[user]:
            if item not in item_users:
                item_users[item] = []
            item_users[item].append(user)

    # 计算用户相似度矩阵
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i: continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                # 相比UserCF，主要是改进了这里,同一个item，感兴趣的人越多，说明比较热门，相似度权重要小，越冷门的item的user相似度越高
                sim[u][v] += 1 / math.log(1 + len(users))
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True))
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for u, _ in sorted_user_sim[user][:K]:
            for item in train[u]:
                # 要去掉用户见过的
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation

# 测试
class Experiment():

    def __init__(self, M, K, N, fp='E:\PythonWorkSpace\pycharm\data\movies_data\\ratings.dat', rt='UserCF'):
        '''
        :params: M, 进行多少次实验
        :params: K, TopK相似用户的个数
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'Random': Random, 'MostPopular': MostPopular,
                    'UserCF': UserCF, 'UserIIF': UserIIF}

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.K, self.N)
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
        print('Average Result (M={}, K={}, N={}): {}'.format(self.M, self.K, self.N, metrics))

