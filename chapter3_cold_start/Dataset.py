

# 导入包
import random
import time


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

    def __init__(self, fp, up):
        # fp: data file path
        # up: user profile path
        self.data, self.profile = self.loadData(fp, up)

    @timmer
    def loadData(self, fp, up):
        data = []
        for l in open(fp,encoding='utf-8'):
            data.append(tuple(l.strip().split('\t')[:2]))
        profile = {}
        for l in open(up,encoding='utf-8'):
            user, gender, age, country, _ = l.strip().split('\t')
            if age == '':
                age = -1
            profile[user] = {'gender': gender, 'age': int(age), 'country': country}
        # 按照用户进行采样
        users = list(profile.keys())
        random.shuffle(users)
        users = set(users[:5000])# 共359347人
        data = [x for x in data if x[0] in users]
        profile = {k: profile[k] for k in users}
        return data, profile

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
            # 取M-1较为合理，因randint是左右都覆盖的
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

        return convert_dict(train), convert_dict(test), self.profile