# 导入包
import math
import time
from code_w.recommand.chapter4.database import Dataset
from code_w.recommand.chapter4.metric import Metric

# 定义装饰器，监控运行时间
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper

# 1. 基于热门标签的推荐
def SimpleTagBased(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计user_tags和tag_items
    user_tags, tag_items = {}, {}
    for user in train:
        user_tags[user] = {}
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                if tag not in tag_items:
                    tag_items[tag] = {}
                if item not in tag_items[tag]:
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1

    def GetRecommendation(user):
        # 按照打分推荐N个未见过的
        if user not in user_tags:
            return []
        seen_items = set(train[user])
        item_score = {}
        for tag in user_tags[user]:
            for item in tag_items[tag]:
                if item in seen_items:
                    continue
                if item not in item_score:
                    item_score[item] = 0
                item_score[item] += user_tags[user][tag] * tag_items[tag][item]
        item_score = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))
        return item_score[:N]

    return GetRecommendation


# 2. 改进一：为热门标签加入惩罚项
def TagBasedTFIDF(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计user_tags和tag_items
    user_tags, tag_items = {}, {}
    # 统计标签的热门程度，即打过此标签的不同用户数
    tag_pop = {}
    for user in train:
        user_tags[user] = {}
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                if tag not in tag_items:
                    tag_items[tag] = {}
                if item not in tag_items[tag]:
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1
                if tag not in tag_pop:
                    tag_pop[tag] = set()
                tag_pop[tag].add(user)
    tag_pop = {k: len(v) for k, v in tag_pop.items()}
    # tag_pop = {k: math.log(1 + len(v))for k, v in tag_pop.items()}

    def GetRecommendation(user):
        # 按照打分推荐N个未见过的
        if user not in user_tags:
            return []
        seen_items = set(train[user])
        item_score = {}
        for tag in user_tags[user]:
            for item in tag_items[tag]:
                if item in seen_items:
                    continue
                if item not in item_score:
                    item_score[item] = 0
                item_score[item] += user_tags[user][tag] * tag_items[tag][item] / tag_pop[tag]
        item_score = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))
        return item_score[:N]

    return GetRecommendation


# 3. 改进二：同时也为热门商品加入惩罚项
def TagBasedTFIDF_Improved(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计user_tags和tag_items
    user_tags, tag_items = {}, {}
    # 统计标签和物品的热门程度，即打过此标签的不同用户数，和物品对应的不同用户数
    tag_pop, item_pop = {}, {}
    for user in train:
        user_tags[user] = {}
        for item in train[user]:
            if item not in item_pop:
                item_pop[item] = 0
            item_pop[item] += 1
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                if tag not in tag_items:
                    tag_items[tag] = {}
                if item not in tag_items[tag]:
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1
                if tag not in tag_pop:
                    tag_pop[tag] = set()
                tag_pop[tag].add(user)
    tag_pop = {k: len(v) for k, v in tag_pop.items()}
    # tag_pop = {k: math.log(1+len(v)) for k, v in tag_pop.items()}

    def GetRecommendation(user):
        # 按照打分推荐N个未见过的
        if user not in user_tags:
            return []
        seen_items = set(train[user])
        item_score = {}
        for tag in user_tags[user]:
            for item in tag_items[tag]:
                if item in seen_items:
                    continue
                if item not in item_score:
                    item_score[item] = 0
                # item_score[item] += user_tags[user][tag] * tag_items[tag][item] / tag_pop[tag] / math.log(item_pop[item]+1)
                item_score[item] += user_tags[user][tag] * tag_items[tag][item] / tag_pop[tag] / item_pop[item]
        item_score = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))
        return item_score[:N]

    return GetRecommendation


# 4. 基于标签改进的推荐
def ExpandTagBased(train, N, M=20):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: M，超参数，设置取TopM的标签填补不满M个标签的用户
    :return: GetRecommendation，推荐接口函数
    '''

    # 1. 计算标签之间的相似度
    item_tag = {}
    for user in train:
        for item in train[user]:
            if item not in item_tag:
                item_tag[item] = set()
            for tag in train[user][item]:
                item_tag[item].add(tag)
    tag_sim, tag_cnt = {}, {}
    for item in item_tag:
        for u in item_tag[item]:
            if u not in tag_cnt:
                tag_cnt[u] = 0
            tag_cnt[u] += 1
            if u not in tag_sim:
                tag_sim[u] = {}
            for v in item_tag[item]:
                if u == v:
                    continue
                if v not in tag_sim[u]:
                    tag_sim[u][v] = 0
                tag_sim[u][v] += 1
    for u in tag_sim:
        for v in tag_sim[u]:
            tag_sim[u][v] /= math.sqrt(tag_cnt[u] * tag_cnt[v])

    # 2. 为每个用户扩展标签
    user_tags = {}
    for user in train:
        if user not in user_tags:
            user_tags[user] = {}
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
    expand_tags = {}
    for user in user_tags:
        if len(user_tags[user]) >= M:
            expand_tags[user] = user_tags[user]
            continue
        # 不满M个的进行标签扩展
        expand_tags[user] = {}
        seen_tags = set(user_tags[user])
        for tag in user_tags[user]:
            for t in tag_sim[tag]:
                if t in seen_tags:
                    continue
                if t not in expand_tags[user]:
                    expand_tags[user][t] = 0
                expand_tags[user][t] += user_tags[user][tag] * tag_sim[tag][t]#相关性加权，生成新的tag权值
        expand_tags[user].update(user_tags[user])
        expand_tags[user] = dict(list(sorted(expand_tags[user].items(), key=lambda x: x[1], reverse=True))[:M])

    # 3. SimpleTagBased算法
    tag_items = {}
    for user in train:
        for item in train[user]:
            for tag in train[user][item]:
                if tag not in tag_items:
                    tag_items[tag] = {}
                if item not in tag_items[tag]:
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1

    def GetRecommendation(user):
        # 按照打分推荐N个未见过的
        if user not in user_tags:
            return []
        seen_items = set(train[user])
        item_score = {}
        for tag in expand_tags[user]:
            for item in tag_items[tag]:
                if item in seen_items:
                    continue
                if item not in item_score:
                    item_score[item] = 0
                item_score[item] += expand_tags[user][tag] * tag_items[tag][item]
        item_score = list(sorted(item_score.items(), key=lambda x: x[1], reverse=True))
        return item_score[:N]

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, fp='../../../data/hetrec2011-delicious-2k/user_taggedbookmarks.dat', rt='SimpleTagBased'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'SimpleTagBased': SimpleTagBased, 'TagBasedTFIDF': TagBasedTFIDF, \
                    'TagBasedTFIDF_Improved': TagBasedTFIDF_Improved, 'ExtendTagBased': ExpandTagBased}

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Diversity': 0,
                   'Popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}): {}'.format(self.M, self.N, metrics))


# # 1. SimpleTagBased实验
# M, N = 10, 10
# exp = Experiment(M, N, rt='SimpleTagBased')
# exp.run()

# 2. TagBasedTFIDF实验
# M, N = 10, 10
# exp = Experiment(M, N, rt='TagBasedTFIDF')
# exp.run()


# 3. TagBasedTFIDF++实验
M, N = 10, 10
exp = Experiment(M, N, rt='TagBasedTFIDF_Improved')
exp.run()

# # 4. TagExtend实验
# M, N = 10, 10
# exp = Experiment(M, N, rt='ExtendTagBased')
# exp.run()

