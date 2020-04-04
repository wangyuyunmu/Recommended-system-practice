from code_w.recommand.chapter2.train_itemCFs import Experiment


# 1. ItemCF实验
M, N = 8, 10
for K in [5, 10, 20, 40, 80, 160]:
    cf_exp = Experiment(M, K, N, rt='ItemCF')
    cf_exp.run()