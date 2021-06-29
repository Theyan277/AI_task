import copy
import Genetic
from GA import cal_pop_fitness, select_mating_pool
import GA as ga
import numpy as np
import random
#
# geneNum = 8  # 构造8个基因，对应8个个体
# geneSize = 4  # 每个基因的大小为4
#
# # 基因设置信息
# gene_settings = (geneNum, geneSize)
#
# # 生成一个个体基因组数组,数组的大小为基因数*单个基因规模,每个基因的数值是3-9的float
# new_population = np.random.uniform(low=3, high=4, size=gene_settings)
#
# # 令new_population为二维数组,第一维表示每个个体?第二维表示每个个体的基因?基因是2的int(float随机数)次方?
# new_population = [[2 ** int(j) for j in i] for i in new_population]
#
# # 又对每个个体的0、3号基因做出了调整,不知道为什么
# for i in range(len(new_population)):
#     new_population[i][0] = random.randint(1, 3)
#     new_population[i][3] = 2 ** random.randint(3, 10)
# new_population = np.array(new_population)
#
# num_generations = 2
#
# num_parents_mating = 4

# 个体数,没有建议,但是之前求方程根的时候都是一万以上
maxPop = 4
initialPop = maxPop

# 遗传算法计算的代数
generation = 50

# 基因大小,不太需要改,如果随便改会出错
geneSize = 20

# def getSolution(indi):
#     list=[]
#     list.append(indi.getGene(0,4),0,5)
#     list.append(indi.getGene(5,9),0,5)
#     list.append(indi.getGene(10,19),0,10)
#     return list

print('开始第0代随机计算')
g = Genetic.GeneticCalculator(8,8,20,ga.create_train_test)
# print('第0代计算完毕,当前最优解:'+str(getSolution(g.currentBest))+',其评估结果:'+str(g.bestEstimate))
print("第0代计算完成")
for i in range(generation):
    print('开始第'+str(i+1)+'代的计算:')
    g.calculate(1)
    print('第'+str(i+1)+'代计算完毕')
#     print('第'+str(i+1)+'代计算完毕,当前最优解:'+str(getSolution(g.currentBest))+',其评估结果:'+str(g.bestEstimate))


# 每一代操作
# for generation in range(num_generations):
#     print("Generation = " + str(generation + 1))
#     # Measuring the fitness of each chromosome in the population.
#     if generation == 0:
#         fitness = cal_pop_fitness(new_population)
#         fitness_copy = sorted(copy.deepcopy(fitness))
#         top = fitness[0].copy()
#     else:
#         fitness[:4] = fitness_copy[:4]
#         fitness[4:] = cal_pop_fitness(new_population[4:, :])
#         top = fitness[0].copy()
#     # Selecting the best parents in the population for mating.
#     parents = select_mating_pool(new_population, fitness,
#                                  num_parents_mating)
#     # Generating next generation using crossover.
#     offspring_crossover = ga.crossover(parents,
#                                        offspring_size=(gene_settings[0] - parents.shape[0], geneSize))
#     # Adding some variations to the offsrping using mutation.
#     offspring_mutation = ga.mutation(offspring_crossover)
#     # Creating the new population based on the parents and offspring.
#     new_population[0:parents.shape[0], :] = parents
#     new_population[parents.shape[0]:, :] = offspring_mutation
#
# print(new_population[0] + top)
