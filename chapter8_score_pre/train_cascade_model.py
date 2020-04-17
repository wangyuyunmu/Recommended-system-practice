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


class Data():

    def __init__(self, user, item, rate, test=False, predict=0.0):
        self.user = user
        self.item = item
        self.rate = rate
        self.test = test
        self.predict = predict


class Dataset():

    def __init__(self, fp):
        # fp: data file path
        self.data = self.loadData(fp)

    def loadData(self, fp):
        data = []
        for l in open(fp):
            data.append(tuple(map(int, l.strip().split('::')[:3])))
        data = [Data(*d) for d in data]
        return data

    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        random.seed(seed)
        for i in range(len(self.data)):
            # 这里与书中的不一致，本人认为取M-1较为合理，因randint是左右都覆盖的
            if random.randint(0, M - 1) == k:
                self.data[i].test = True

def RMSE(records):
    rmse = {'train_rmse': [], 'test_rmse': []}
    for r in records:
        if r.test:
            rmse['test_rmse'].append((r.rate - r.predict) ** 2)
        else:
            rmse['train_rmse'].append((r.rate - r.predict) ** 2)
    rmse = {'train_rmse': math.sqrt(sum(rmse['train_rmse']) / len(rmse['train_rmse'])),
            'test_rmse': math.sqrt(sum(rmse['test_rmse']) / len(rmse['test_rmse']))}
    return rmse


# 1. Cluster
class Cluster:

    def __init__(self, records):
        self.group = {}

    def GetGroup(self, i):
        return 0


# 2. IdCluster
class IdCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)

    def GetGroup(self, i):
        return i


# 3. UserActivityCluster
class UserActivityCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        activity = {}
        for r in records:
            if r.test: continue
            if r.user not in activity:
                activity[r.user] = 0
            activity[r.user] += 1
        # 按照用户活跃度进行分组
        k = 0
        for user, n in sorted(activity.items(), key=lambda x: x[-1], reverse=False):
            c = int((k * 5) / len(activity))
            self.group[user] = c
            k += 1

    def GetGroup(self, uid):
        if uid not in self.group:
            return -1
        else:
            return self.group[uid]


# 3. ItemPopularityCluster
class ItemPopularityCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        popularity = {}
        for r in records:
            if r.test: continue
            if r.item not in popularity:
                popularity[r.item] = 0
            popularity[r.item] += 1
        # 按照物品流行度进行分组
        k = 0
        for item, n in sorted(popularity.items(), key=lambda x: x[-1], reverse=False):
            c = int((k * 5) / len(popularity))
            self.group[item] = c
            k += 1

    def GetGroup(self, iid):
        if iid not in self.group:
            return -1
        else:
            return self.group[iid]


# 4. UserVoteCluster
class UserVoteCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        vote, cnt = {}, {}
        for r in records:
            if r.test: continue
            if r.user not in vote:
                vote[r.user] = 0
                cnt[r.user] = 0
            vote[r.user] += r.rate
            cnt[r.user] += 1
        # 按照物品平均评分进行分组
        for user, v in vote.items():
            c = v / (cnt[user] * 1.0)
            self.group[user] = int(c * 2)

    def GetGroup(self, uid):
        if uid not in self.group:
            return -1
        else:
            return self.group[uid]


# 5. ItemVoteCluster
class ItemVoteCluster(Cluster):

    def __init__(self, records):
        Cluster.__init__(self, records)
        vote, cnt = {}, {}
        for r in records:
            if r.test: continue
            if r.item not in vote:
                vote[r.item] = 0
                cnt[r.item] = 0
            vote[r.item] += r.rate
            cnt[r.item] += 1
        # 按照物品平均评分进行分组
        for item, v in vote.items():
            c = v / (cnt[item] * 1.0)
            self.group[item] = int(c * 2)

    def GetGroup(self, iid):
        if iid not in self.group:
            return -1
        else:
            return self.group[iid]

# 返回预测接口函数
def PredictAll(records, UserGroup, ItemGroup):
    '''
    :params: records, 数据集
    :params: UserGroup, 用户分组类
    :params: ItemGroup, 物品分组类
    '''
    userGroup = UserGroup(records)
    itemGroup = ItemGroup(records)
    group = {}
    for r in records:
        ug = userGroup.GetGroup(r.user)
        ig = itemGroup.GetGroup(r.item)
        if ug not in group:
            group[ug] = {}
        if ig not in group[ug]:
            group[ug][ig] = []
        # 这里计算的残差
        group[ug][ig].append(r.rate - r.predict)
    for ug in group:
        for ig in group[ug]:
            group[ug][ig] = sum(group[ug][ig]) / (1.0 * len(group[ug][ig]) + 1.0)
    # predict
    for i in range(len(records)):
        ug = userGroup.GetGroup(records[i].user)
        ig = itemGroup.GetGroup(records[i].item)
        # 这里需要与之前的结果进行结合,这里应该用回归训练的方法计算出每个推荐方法的权值。
        records[i].predict += group[ug][ig]


class Experiment():

    def __init__(self, M, UserGroup, ItemGroup, fp='../../../data/movies_data/ratings.dat'):
        '''
        :params: M, 进行多少次实验
        :params: UserGroup, ItemGroup, 聚类算法类型
        :params: fp, 数据文件路径
        '''
        self.userGroup = UserGroup
        self.itemGroup = ItemGroup
        self.dataset = Dataset(fp)
        self.dataset.splitData(M, 0)

    # 定义单次实验
    def worker(self, records):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: train和test的rmse值
        '''
        PredictAll(records, self.userGroup, self.itemGroup)
        metric = RMSE(records)
        return metric

    # 多次实验取平均
    def run(self):
        metrics = {'train_rmse': 0, 'test_rmse': 0}
        metric = self.worker(self.dataset.data)
        print('Result (UserGroup={}, ItemGroup={}): {}'.format(
            self.userGroup.__name__,
            self.itemGroup.__name__, metric))

UserGroups = [Cluster, IdCluster, Cluster, UserActivityCluster, UserActivityCluster, Cluster, IdCluster,
              UserActivityCluster, UserVoteCluster, UserVoteCluster, Cluster, IdCluster, UserVoteCluster]
ItemGroups = [Cluster, Cluster, IdCluster, Cluster, IdCluster, ItemPopularityCluster, ItemPopularityCluster,
              ItemPopularityCluster, Cluster, IdCluster, ItemVoteCluster, ItemVoteCluster, ItemVoteCluster]
M = 10
exp = Experiment(M, None, None)
for i in range(len(UserGroups)):
    exp.userGroup = UserGroups[i]
    exp.itemGroup = ItemGroups[i]
    exp.run()



