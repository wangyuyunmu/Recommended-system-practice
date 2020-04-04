from code_w.recommand.chapter2.train_itemCFs import Experiment

# 3. ItemCF-Norm实验
M, N = 8, 10
K = 10 # 与书中保持一致
norm_exp = Experiment(M, K, N, rt='ItemCF-Norm')
norm_exp.run()