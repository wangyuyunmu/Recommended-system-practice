from code_w.recommand.chapter2.train_userCFs import Experiment

# 3. UserCF实验
M, N = 8, 10
for K in [5, 10, 20, 40, 80, 160]:
    cf_exp = Experiment(M, K, N, rt='UserCF')
    cf_exp.run()