
# 导入包
import random
import math
import numpy as np
import time
from tqdm import tqdm, trange


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

    def __init__(self, fp, ip):
        # fp: data file path
        self.data, self.content = self.loadData(fp, ip)

    @timmer
    def loadData(self, fp, ip):
        data = []
        for l in open(fp):
            data.append(tuple(map(int, l.strip().split('::')[:2])))
        contents = {}
        for l in open(ip, 'rb'):
            l = l.strip()
            l = str(l)[2:-1]
            contents[int(l.strip().split('::')[0])] = l.strip().split('::')[-1].split('|')
        return data, contents

    @timmer
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
        for user, item in self.data:
            # 这里与书中的不一致，本人认为取M-1较为合理，因randint是左右都覆盖的
            if random.randint(0, M - 1) == k:
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test), self.content


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
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2)

    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            all += len(test_items)
        return round(hit / all * 100, 2)

    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test:
            for item in self.train[user]:
                all_item.add(item)
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item) * 100, 2)

    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1

        num, pop = 0, 0
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                if item in item_pop:
                    # 取对数，防止因长尾问题带来的被流行物品所主导
                    pop += math.log(1 + item_pop[item])
                    num += 1
        return round(pop / num, 6)

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        print('Metric:', metric)
        return metric


def ContentItemKNN(train, content, K, N):
    '''
    :params: train, 训练数据
    :params: content, 物品内容信息
    :params: K, 取相似Top-K相似物品
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''

    # 建立word-item倒排表
    word_item = {}
    for item in content:
        for word in content[item]:
            if word not in word_item:
                word_item[word] = {}
            word_item[word][item] = 1#物品item与关键词world的倒排索引world-item

    for word in word_item:
        for item in word_item[word]:
            word_item[word][item] /= math.log(1 + len(word_item[word]))

    # 计算相似度
    item_sim = {}
    mo = {}
    for word in word_item:
        for u in word_item[word]:
            if u not in item_sim:
                item_sim[u] = {}
                mo[u] = 0
            mo[u] += word_item[word][u] ** 2
            for v in word_item[word]:
                if u == v: continue
                if v not in item_sim[u]:
                    item_sim[u][v] = 0
                item_sim[u][v] += word_item[word][u] * word_item[word][v] #每个物品与其他物品的相关性
    for u in item_sim:
        for v in item_sim[u]:
            item_sim[u][v] /= math.sqrt(mo[u] * mo[v])

    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True)) \
                       for k, v in item_sim.items()}#对每个物品，与之相关的物品进行排序

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item in train[user]:
            for u, _ in sorted_item_sim[item][:K]:
                # 要去掉用户见过的
                if u not in seen_items:
                    if u not in items:
                        items[u] = 0
                    items[u] += item_sim[item][u] #权值
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, K, fp='../../../data/movies_data/ratings.dat', ip='../../../data/movies_data/movies.dat'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: K, 取Top-K相似物品数目
        :params: fp, 数据文件路径
        :params: ip, 物品内容路径
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.ip = ip
        self.alg = ContentItemKNN

    # 定义单次实验
    @timmer
    def worker(self, train, test, content):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg(train, content, self.K, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp, self.ip)
        for ii in range(self.M):
            train, test, content = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test, content)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, K={}): {}'.format(self.M, self.N, self.K, metrics))

M, N, K = 8, 10, 10
exp = Experiment(M, N, K)
exp.run()