# 导入包
import random
import math
import time
from tqdm import tqdm

# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper


class Dataset():

    def __init__(self, fp, sample=100000):
        # fp: data file path
        # sample: 只取部分数据集，-1则为全部
        self.data = self.loadData(fp, sample)

    def loadData(self, fp, sample):
        # 只取一个小数据集进行处理
        data = [f.strip().split('\t') for f in open(fp).readlines()[4:]]
        if sample == -1:
            return data
        else:
            random.shuffle(data)
            return data[:sample]

    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        train, test = [], []
        random.seed(seed)
        for u, v in self.data:
            # 这里与书中的不一致，本人认为取M-1较为合理，因randint是左右都覆盖的
            if random.randint(0, M - 1) == k:
                test.append((u, v))
            else:
                train.append((u, v))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}  # 当前用户指向的用户
            data_dict_t = {}  # 指向当前用户的用户
            for u, v in data:
                if u not in data_dict:
                    data_dict[u] = set()
                data_dict[u].add(v)
                if v not in data_dict_t:
                    data_dict_t[v] = set()
                data_dict_t[v].add(u)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            data_dict_t = {k: list(data_dict_t[k]) for k in data_dict_t}
            return data_dict, data_dict_t

        return convert_dict(train), convert_dict(test)[0]


class Metric():

    def __init__(self, train, test, GetRecommendation):
        '''
        :params: train, 训练数据
        :params: test, 测试数据
        :params: GetRecommendation, 为某个用户获取推荐物品的接口函数
        '''
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommendation(user)
            recs[user] = rank
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        all, hit = 0, 0
        for user in self.test:
            test_users = set(self.test[user])
            rank = self.recs[user]
            for v, score in rank:
                if v in test_users:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2) if all > 0 else 0

    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test:
            test_users = set(self.test[user])
            rank = self.recs[user]
            for v, score in rank:
                if v in test_users:
                    hit += 1
            all += len(test_users)
        return round(hit / all * 100, 2) if all > 0 else 0

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall()}
        print('Metric:', metric)
        return metric


# 1. 利用用户出度计算相似度
def OUT(train, N):
    '''
    :params: train, 训练数据集（包含出和入的）
    :params: N, 超参数，设置取TopN推荐用户数目
    :return: GetRecommendation，推荐接口函数
    '''

    G, GT = train # 分别为out和in

    def GetRecommendation(user):
        if user not in G: return []
        # 根据相似度推荐N个未见过的
        user_sim = {}
        user_friends = set(G[user])
        for u in G[user]:
            if u not in GT: continue
            for v in GT[u]:# 如果u的朋友是x（out[u]=x），寻找相似user v，v的好友也是x(out[v]=x)，那么搜索in[x]=v
                if v != user and v not in user_friends:
                    if v not in user_sim:
                        user_sim[v] = 0
                    user_sim[v] += 1
        user_sim = {v: user_sim[v] / math.sqrt(len(G[user]) * len(G[v])) for v in user_sim}
        return list(sorted(user_sim.items(), key=lambda x: x[1], reverse=True))[:N]

    return GetRecommendation


# 2. 利用用户入度计算相似度
def IN(train, N):
    '''
    :params: train, 训练数据集（包含出和入的）
    :params: N, 超参数，设置取TopN推荐用户数目
    :return: GetRecommendation，推荐接口函数
    '''

    G, GT = train

    def GetRecommendation(user):
        if user not in GT: return []
        # 根据相似度推荐N个未见过的
        user_sim = {}
        user_friends = set(G[user]) if user in G else set()
        for u in GT[user]:
            if u not in G: continue
            for v in G[u]:
                if v != user and v not in user_friends:
                    if v not in user_sim:
                        user_sim[v] = 0
                    user_sim[v] += 1
        user_sim = {v: user_sim[v] / math.sqrt(len(GT[user] * len(GT[v]))) for v in user_sim}
        return list(sorted(user_sim.items(), key=lambda x: x[1], reverse=True))[:N]

    return GetRecommendation


# 3. 利用用户出度和入度进行计算，但没有考虑到热门入度用户的惩罚
def OUT_IN(train, N):
    '''
    :params: train, 训练数据集（包含出和入的）
    :params: N, 超参数，设置取TopN推荐用户数目
    :return: GetRecommendation，推荐接口函数
    '''

    G, GT = train

    def GetRecommendation(user):
        if user not in G: return []
        # 根据相似度推荐N个未见过的
        user_sim = {}
        user_friends = set(G[user])
        for u in G[user]:
            if u not in G: continue
            for v in G[u]:
                if v != user and v not in user_friends:
                    if v not in user_sim:
                        user_sim[v] = 0
                    user_sim[v] += 1
        user_sim = {v: user_sim[v] / len(G[user]) for v in user_sim}
        return list(sorted(user_sim.items(), key=lambda x: x[1], reverse=True))[:N]

    return GetRecommendation


# 4. 利用用户出度和入度的余弦相似度进行计算
def OUT_IN_Cosine(train, N):
    '''
    :params: train, 训练数据集（包含出和入的）
    :params: N, 超参数，设置取TopN推荐用户数目
    :return: GetRecommendation，推荐接口函数
    '''

    G, GT = train

    def GetRecommendation(user):
        if user not in G: return []
        # 根据相似度推荐N个未见过的
        user_sim = {}
        user_friends = set(G[user])
        for u in G[user]:
            if u not in G: continue
            for v in G[u]:
                if v != user and v not in user_friends:
                    if v not in user_sim:
                        user_sim[v] = 0
                    user_sim[v] += 1
        user_sim = {v: user_sim[v] / math.sqrt(len(G[user]) * len(GT[v])) for v in user_sim}
        return list(sorted(user_sim.items(), key=lambda x: x[1], reverse=True))[:N]

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, fp='../../../data/soc-Slashdot0902', rt='OUT'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐用户的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'OUT': OUT, 'IN': IN,
                    'OUT_IN': OUT_IN, 'OUT_IN_Cosine': OUT_IN_Cosine}

    # 定义单次实验
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.N)
        metric = Metric(train[0], test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, alg={}): {}'.format(
            self.M, self.N, self.rt, metrics))

# 1. Slashdot数据集实验
M, N = 10, 10
for alg in ['OUT', 'IN', 'OUT_IN', 'OUT_IN_Cosine']:
    exp = Experiment(M, N, fp='../../../data/soc-Slashdot0902/Slashdot0902.txt', rt=alg)
    exp.run()


# 2. Epinions数据集实验
M, N = 10, 10
for alg in ['OUT', 'IN', 'OUT_IN', 'OUT_IN_Cosine']:
    exp = Experiment(M, N, fp='../../../data/soc-Epinions1/soc-Epinions1.txt', rt=alg)
    exp.run()

