# 导入包
import codecs

class Dataset():

    # 对每个用户按照时间进行从前到后的排序，取最后一个时间的item作为要预测的测试集

    def __init__(self, site=None):
        # site: which site to load
        self.bookmark_path = '../../../data/hetrec2011-delicious-2k/bookmarks.dat'
        self.user_bookmark_path = '../../../data/hetrec2011-delicious-2k/user_taggedbookmarks-timestamps.dat'
        self.site = site
        self.loadData()

    def loadData(self):
        bookmarks = [f.strip() for f in codecs.open(self.bookmark_path, 'r', encoding="ISO-8859-1").readlines()][1:]
        # 将不同网站对应的item与时间保存下来
        site_ids = {}
        for b in bookmarks:
            b = b.split('\t')
            if b[-1] not in site_ids:
                site_ids[b[-1]] = set()
            site_ids[b[-1]].add(b[0])

        user_bookmarks = [f.strip() for f in
                          codecs.open(self.user_bookmark_path, 'r', encoding="ISO-8859-1").readlines()][1:]
        data = {}
        cnt = 0
        # 选择和收集对应的url的user:item+time数据，并按照时间对每个user进行排序
        for ub in user_bookmarks:
            ub = ub.split('\t')
            if self.site is None or (self.site in site_ids and ub[1] in site_ids[self.site]):
                if ub[0] not in data:
                    data[ub[0]] = set()
                data[ub[0]].add((ub[1], int(ub[3][:-3])))
                cnt += 1
        self.data = {k: list(sorted(list(data[k]), key=lambda x: x[1], reverse=True)) for k in data}

    def splitData(self):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :return: train, test
        '''
        train, test = {}, {}
        for user in self.data:
            if user not in train:
                train[user] = []
                test[user] = []
            data = self.data[user]
            train[user].extend(data[1:])#
            test[user].append(data[0])# 时间最大作为测试

        return train, test