import numpy as np
import random
from tutil import choose2, getpop, mergeGenes,vary, getfit, cross2
geneNum = 100  # 种群数量
generationNum = 300  # 迭代次数

CENTER = 0  # 配送中心

# HUGE = 9999999
# PC = 1   #交叉率,没有定义交叉率，也就是说全部都要交叉，也就是1
PM = 0.1  # 变异率   以前是用的vary

n = 25  # 客户点数量
m = 2  # 换电站数量
k = 3  # 车辆数量
Q = 5  # 额定载重量, t
# dis = 160  # 续航里程, km
length = n+m+1

# 坐标   第0个是配送中心   1-25是顾客      26和27是换电站          一共28个位置    行驶距离要通过这个坐标自己来算
X = [56, 66, 56, 88, 88, 24, 40, 32, 16, 88, 48, 32, 80, 48, 23, 48, 16, 8, 32, 24, 72, 72, 72, 88, 104, 104, 83,32]
Y = [56, 78, 27, 72, 32, 48, 48, 80, 69, 96, 96, 104, 56, 40, 16, 8, 32, 48, 64, 96, 104, 32, 16, 8, 56, 32, 45, 40]
# 需求量
t = [0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8, 0.4, 0.5, 0.7, 0.7, 0.6, 0.2, 0.2, 0.4, 0.1, 0.1, 0.2, 0.5, 0.2, 0.7,0.2,0.7, 0.1, 0.5, 0.4, 0.4]


if __name__ == '__main__':
    import numpy as np
    import random
    from tqdm import *  # 进度条
    import matplotlib.pyplot as plt
    from pylab import *

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    best_fitness = []
    min_cost = []
    J = []
    pop = getpop(length, geneNum)  # 初始种群
    # 迭代
    for j in tqdm(range(generationNum)):
        print('j=', j)
        chosen_pop = choose2(pop)  # 选择   选择适应度值最高的前三分之一，也就是32个种群，进行下一步的交叉
        crossed_pop = cross2(chosen_pop)  # 交叉
        pop = mergeGenes(pop, crossed_pop)  # 复制交叉至子代种群
        pop = vary(pop)  # under construction
        key = lambda gene: getfit(gene)
        pop.sort(reverse=True, key=key)  # 以fit对种群排序
        cost = 1 / getfit(pop[0])
        print(cost)
        min_cost.append(cost)
        J.append(j)
    print(J)
    print(min_cost)

    # key = lambda gene: getfit(gene)
    # pop.sort(reverse=True, key=key)   # 以fit对种群排序
    print('\r\n')
    print('data:', pop[0])
    print('fit:', 1 / getfit(pop[0]))
    plt.plot(J, min_cost, color='r')
    plt.show()


