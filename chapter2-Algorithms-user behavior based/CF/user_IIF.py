from code_w.recommand.chapter2.train_userCFs import Experiment

# 4. UserIIF实验
M, N = 8, 10
K = 80 # 与书中保持一致
iif_exp = Experiment(M, K, N, rt='UserIIF')
iif_exp.run()