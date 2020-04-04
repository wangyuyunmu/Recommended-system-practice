from code_w.recommand.chapter2.train_itemCFs import Experiment


# 2. ItemIUF实验
M, N = 8, 10
K = 10 # 与书中保持一致
iuf_exp = Experiment(M, K, N, rt='ItemIUF')
iuf_exp.run()