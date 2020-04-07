# 导入包
import random
import math
import numpy as np
import time
from tqdm import tqdm
from scipy.sparse import csc_matrix, linalg, eye
from copy import deepcopy
from code_w.recommand.chapter2.graph_based.Dataset import Dataset
from code_w.recommand.chapter2.graph_based.metrics import Metric

# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper


def PersonalRank(train, alpha, N):
    '''
    :params: train, 训练数据
    :params: alpha, 继续随机游走的概率
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''

    # 构建索引
    items = []
    for user in train:
        items.extend(train[user])
    id2item = list(set(items))                                #item集合
    users = {u: i for i, u in enumerate(train.keys())}        #user编号
    items = {u: i + len(users) for i, u in enumerate(id2item)}#item编号，在user之后

    # 计算转移矩阵（注意！！！要按照出度进行归一化）
    item_user = {}
    for user in train:
        for item in train[user]:
            if item not in item_user:
                item_user[item] = []
            item_user[item].append(user)  #item-user倒排索引

    data, row, col = [], [], []
    for u in train:
        for v in train[u]:
            data.append(1 / len(train[u]))# 保存所有节点出度
            row.append(users[u])          # 出度的节点
            col.append(items[v])          # 连接的节点
    for u in item_user:                   # user遍历完之后，再次遍历item
        for v in item_user[u]:
            data.append(1 / len(item_user[u]))
            row.append(items[u])
            col.append(users[v])
    # 行程稀疏矩阵，按照列排列row和col分别代表data位置的索引，shape不太赞同，我觉得应该是len(users)+len(items)而不是len(data)
    M = csc_matrix((data, (row, col)), shape=(len(users)+len(items),len(users)+len(items)))

    # 获取接口函数
    def GetRecommendation(user):
        seen_items = set(train[user])
        # 解矩阵方程 r = (1-a)r0 + a(M.T)r
        # r0 = [0] * (len(users)+len(items))
        r0 = [[0] for i in range(len(users)+len(items))]
        r0[users[user]][0] = 1 #测试那个user就将该user设置为1，表示从此开始随机游走
        r0 = np.array(r0)
        # r0 = csc_matrix(r0) #list转化成稀疏矩阵，按照列排列
        # r = (1 - alpha) * linalg.inv(eye(len(users)+len(items)) - alpha * M.T) * r0 #M是按照列排列的，转置
        # r = r.T.toarray()[0][len(users):]# user 之后的节点才是item

        r = linalg.gmres(eye(len(users) + len(items)) - alpha * M.T, (1 - alpha) * r0)  # gmres(A,b),解决稀疏Ax=b的求解问题，
        r = r[0][len(users):]  # user 之后的节点才是item

        idx = np.argsort(-r)[:N]         # 取反是为了从大到小排列
        recs = [(id2item[ii], r[ii]) for ii in idx]  #返回topN的item与PR值的tuple
        return recs

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, alpha, fp='E:\PythonWorkSpace\pycharm\data\movies_data\\ratings.dat'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: alpha, 继续随机游走的概率
        :params: fp, 数据文件路径
        '''
        self.M = M
        self.N = N
        self.alpha = alpha
        self.fp = fp
        self.alg = PersonalRank

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg(train, self.alpha, self.N)
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
        print('Average Result (M={}, N={}, ratio={}): {}'.format( self.M, self.N, self.ratio, metrics))

# PersonalRank实验(笔记本跑的太慢，这里没贴实验结果)
M, N, alpha = 8, 10, 0.8
exp = Experiment(M, N, alpha)
exp.run()

