# coding=utf-8

def PersonalRank(G, alpha, root, max_depth):
    rank = dict()
    rank = {x: 0 for x in G.keys()}
    rank[root] = 1
    for k in range(max_depth):
        tmp = {x: 0 for x in G.keys()}
        # 取出节点i和他的出边尾节点集合ri
        for i, ri in G.items():
            # 取节点i的出边的尾节点j以及边E(i,j)的权重wij,边的权重都为1，归一化后就是1/len(ri)
            for j, wij in ri.items():
                # 这里可以看出前一个step（k）生成的图每个节点以alpha概率向其他相关节点传递PR值，
                # 生成新的图，但是每个节点都有1-alpha概率保留PR，所以新图整体少了1-alpha
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))
        tmp[root] += (1 - alpha)
        rank = tmp
    lst = sorted(rank.items(), key=lambda x: x[1], reverse=True)
    for ele in lst:
        print("%s:%.3f, \t" % (ele[0], ele[1]))
    return rank


if __name__ == '__main__':
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}
    PersonalRank(G, 0.85, 'A', 100)