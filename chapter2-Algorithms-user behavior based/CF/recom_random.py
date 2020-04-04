
from code_w.recommand.chapter2.train_userCFs import Experiment

# 1. random实验
M, N = 8, 10
K = 0 # 为保持一致而设置，随便填一个值
random_exp = Experiment(M, K, N, rt='Random')
random_exp.run()