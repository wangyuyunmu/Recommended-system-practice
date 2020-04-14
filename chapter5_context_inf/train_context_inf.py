# 导入包
import math
import time

from code_w.recommand.chapter5.Dataset import Dataset
from code_w.recommand.chapter5.Metric import Metric


# 1. 给用户推荐近期最热门的物品
def RecentPopular(train, K, N, alpha=1.0, t0=int(time.time())):
    '''
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: alpha, 时间衰减因子
    :params: t0, 当前的时间戳
    :return: GetRecommendation，推荐接口函数
    '''

    item_score = {}
    for user in train:
        for item, t in train[user]:
            if item not in item_score:
                item_score[item] = 0
            item_score[item] += 1.0 / (alpha * (t0 - t))

    item_score = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))

    def GetRecommendation(user):
        # 随机推荐N个未见过的
        user_items = set(train[user])
        rec_items = [x for x in item_score if x[0] not in user_items]
        return rec_items[:N]

    return GetRecommendation


# 2. 时间上下文相关的ItemCF算法
def TItemCF(train, K, N, alpha=1.0, beta=1.0, t0=int(time.time())):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: alpha, 计算item相似度的时间衰减因子
    :params: beta, 推荐打分时的时间衰减因子
    :params: t0, 当前的时间戳
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算物品相似度矩阵
    sim = {}
    num = {}
    for user in train:
        items = train[user]
        for i in range(len(items)):
            u, t1 = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if j == i: continue
                v, t2 = items[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1.0 / (alpha * (abs(t1 - t2) + 1))
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) \
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item, t in train[user]:
            for u, _ in sorted_item_sim[item][:K]:
                if u not in seen_items:
                    if u not in items:
                        items[u] = 0
                    items[u] += sim[item][u] / (1 + beta * (t0 - t))
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


# 3. 时间上下文相关的UserCF算法
def TUserCF(train, K, N, alpha=1.0, beta=1.0, t0=int(time.time())):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似用户数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: alpha, 计算item相似度的时间衰减因子
    :params: beta, 推荐打分时的时间衰减因子
    :params: t0, 当前的时间戳
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算item->user的倒排索引
    item_users = {}
    for user in train:
        for item, t in train[user]:
            if item not in item_users:
                item_users[item] = []
            item_users[item].append((user, t))

    # 计算用户相似度矩阵
    sim = {}
    num = {}
    for item in item_users:
        users = item_users[item]
        for i in range(len(users)):
            u, t1 = users[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if j == i: continue
                v, t2 = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1.0 / (alpha * (abs(t1 - t2) + 1))
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_user_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True)) \
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        recs = []
        if user in sorted_user_sim:
            for u, _ in sorted_user_sim[user][:K]:
                for item, _ in train[u]:
                    if item not in seen_items:
                        if item not in items:
                            items[item] = 0
                        items[item] += sim[user][u] / (1 + beta * (t0 - t))
            recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


# 4. ItemCF算法
def ItemCF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算物品相似度矩阵
    sim = {}
    num = {}
    for user in train:
        items = train[user]
        for i in range(len(items)):
            u, _ = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if j == i: continue
                v, _ = items[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按照相似度排序
    sorted_item_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True)) \
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        for item, _ in train[user]:
            for u, _ in sorted_item_sim[item][:K]:
                if u not in seen_items:
                    if u not in items:
                        items[u] = 0
                    items[u] += sim[item][u]
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


# 5. UserCF算法
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
        for item, _ in train[user]:
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
    sorted_user_sim = {k: list(sorted(v.items(),key=lambda x: x[1], reverse=True)) \
                       for k, v in sim.items()}

    # 获取接口函数
    def GetRecommendation(user):
        items = {}
        seen_items = set(train[user])
        recs = []
        if user in sorted_user_sim:
            for u, _ in sorted_user_sim[user][:K]:
                for item, _ in train[u]:
                    # 要去掉用户见过的
                    if item not in seen_items:
                        if item not in items:
                            items[item] = 0
                        items[item] += sim[user][u]
            recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


class Experiment():

    def __init__(self, K, N, site=None, rt='RecentPopular'):
        '''
        :params: K, TopK相似的个数
        :params: N, TopN推荐物品的个数
        :params: site, 选择一个网站的记录进行推荐
        :params: rt, 推荐算法类型
        '''
        self.K = K
        self.N = N
        self.site = site
        self.rt = rt
        self.alg = {'RecentPopular': RecentPopular, 'TItemCF': TItemCF,
                    'TUserCF': TUserCF, 'ItemCF': ItemCF, 'UserCF': UserCF}

    # 定义单次实验
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.K, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 运行实验
    def run(self):
        dataset = Dataset(self.site)
        train, test = dataset.splitData()
        metric = self.worker(train, test)
        print('Result (site={}, K={}, N={}): {}'.format(
            self.site, self.K, self.N, metric))

# 1. RecentPopular实验
K = 0 # 为保持一致而设置，随便填一个值
for site in ['www.nytimes.com', 'en.wikipedia.org']:
    for N in range(10, 110, 10):
        exp = Experiment(K, N, site=site, rt='RecentPopular')
        exp.run()

# 2. TItemCF实验
K = 10
for site in ['www.nytimes.com', 'en.wikipedia.org']:
    for N in range(10, 110, 10):
        exp = Experiment(K, N, site=site, rt='TItemCF')
        exp.run()

# 3. TUserCF实验
K = 10
for site in ['www.nytimes.com', 'en.wikipedia.org']:
    for N in range(10, 110, 10):
        exp = Experiment(K, N, site=site, rt='TUserCF')
        exp.run()

# 4. ItemCF实验
K = 10
for site in ['www.nytimes.com', 'en.wikipedia.org']:
    for N in range(10, 110, 10):
        exp = Experiment(K, N, site=site, rt='ItemCF')
        exp.run()

# 5. UserCF实验
K = 10
for site in ['www.nytimes.com', 'en.wikipedia.org']:
    for N in range(10, 110, 10):
        exp = Experiment(K, N, site=site, rt='UserCF')
        exp.run()

