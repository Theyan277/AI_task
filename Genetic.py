import random
import math

# ranSeed = int(random.random() * 1000000000)
ranSeed = 313094870
random.seed(ranSeed)
print('本次运行的随机种子: ' + str(ranSeed))
print('第一个随机数:' + str(random.random()))


class Individual:
    estimation = 0.0
    size = 0
    gene = []

    def __init__(self, geneSize, randomGene=False):
        if randomGene:
            self.gene = [True if random.random() < 0.5 else False for i in range(geneSize)]
        else:
            self.gene = [False for i in range(geneSize)]
        self.size = geneSize

    def setGene(self, index, val):
        self.gene[index] = val

    def getGene(self, begin=0, end=size - 1):
        return self.gene[begin:end + 1]


class GeneticCalculator:
    # 种群
    population = []
    # 最大个体数
    maxPop = 0
    # 当前个体数
    currentPop = 0
    # 个体基因大小
    geneSize = 0
    # 当前最佳个体
    currentBest = None
    # 当前最佳个体评估值
    bestEstimate = 0.0
    # 淘汰率
    outRate = 0.88
    # 变异率
    variationRate = 0.34
    # 变异力度
    variationStrength = 0.36
    # 种群繁殖优先随机的枢轴权重,大于0的数,越大分布越集中于枢轴处
    growGaussFirstWeight = 2

    # 额外属性:
    # fitnessFunction 适应度计算函数
    # 参数:个体 返回值:个体的适应度,越大越好
    # growFunction 种群增长函数
    # 参数:种群,种群最大个体数 返回值:增长后的种群
    # variationFunction 变异计算函数
    # 参数:种群 返回值:变异后的种群
    # selectionFunction 选择函数
    # 参数:种群 返回值:选择后的种群
    # currentBest 当前最佳个体
    # 类型: Indivisual
    # 种群大小,基因大小,
    def __init__(self, maxPopSize, initialPopSize, geneNum, fitnessFunc, gaussWeight=2):
        self.currentPop = initialPopSize
        self.maxPop = maxPopSize
        self.fitnessFunction = fitnessFunc
        self.growFunction = self.hfGeneEqualProbabilityGrow
        self.variationFunction = self.bvgkVariation
        self.selectionFunction = self.badoutSelection
        self.geneSize = geneNum
        self.growGaussFirstWeight = gaussWeight
        for i in range(initialPopSize):
            self.population.append(Individual(geneNum, True))
        self.estimateAndSortPopulation()

    def populationEstimate(self):
        for i in self.population:
            self.fitnessFunction(i)

    # 高适应度优先 等位基因等概率交换,调用前需要排序个体
    def hfGeneEqualProbabilityGrow(self, population, maxPop):
        currentPop = len(population)
        growNum = maxPop - currentPop
        currentGrow = 0
        growList = []
        while currentGrow < growNum:
            indexA = normalvariateIndex(0, currentPop - 1, currentPop - 1, self.growGaussFirstWeight)
            indexB = normalvariateIndex(0, currentPop - 1, currentPop - 1, self.growGaussFirstWeight)
            indiA = self.population[indexA]
            indiB = self.population[indexB]
            newindi = Individual(self.geneSize, False)
            for i in range(self.geneSize):
                if random.random() < 0.5:
                    newindi.setGene(i, indiA.getGene(i, i)[0])
                else:
                    newindi.setGene(i, indiB.getGene(i, i)[0])
            currentGrow += 1
            growList.append(newindi)
        retList = [population[i] for i in range(currentPop)]
        for i in range(growNum):
            retList.append(growList[i])
        return retList

    # bad variation,good keep
    def bvgkVariation(self, population):
        for i in range(self.currentPop):
            curvariarate = self.variationRate
            curvariastrength = self.variationStrength
            if i < self.currentPop * self.outRate:
                curvariarate += (1 - self.variationRate) * (1 - i / (self.currentPop * self.outRate))
                curvariastrength += (1 - self.variationStrength) * (1 - i / (self.currentPop * self.outRate))
            else:
                curvariarate = self.variationRate * (
                        1 - (i - self.currentPop * self.outRate) / (self.currentPop * (1 - self.outRate)))
                curvariastrength = self.variationStrength * (
                        1 - (i - self.currentPop * self.outRate) / (self.currentPop * (1 - self.outRate)))
            if random.random() < curvariarate:
                for j in range(self.geneSize):
                    population[i].setGene(j,
                                          not population[i].getGene(j, j)[0] if random.random() < curvariastrength else
                                          population[i].getGene(j, j)[0])
        return population

    def badoutSelection(self, population):
        keepIndex = int(self.outRate * self.currentPop)
        return population[keepIndex:]

    def estimateAndSortPopulation(self):
        self.bestEstimate = self.fitnessFunction(self.population[0])
        self.currentBest = self.population[0]
        self.population[0].estimation = self.bestEstimate
        for i in range(1, self.currentPop):
            curestimate = self.fitnessFunction(self.population[i])
            self.population[i].estimation = curestimate
            if curestimate > self.bestEstimate:
                self.bestEstimate = curestimate
                self.currentBest = self.population[i]
        self.population.sort(key=self.fitnessFunction, )

    def calculate(self, generationNum):
        for i in range(generationNum):
            self.estimateAndSortPopulation()
            self.population = self.variationFunction(self.population)
            self.estimateAndSortPopulation()
            afterselection = self.selectionFunction(self.population)
            self.estimateAndSortPopulation()
            self.population = afterselection
            self.currentPop = len(afterselection)
            aftergrow = self.growFunction(self.population, self.maxPop)
            self.population = aftergrow
            self.currentPop = len(aftergrow)


def boolarr2double(arr, ibegin, iend, nbegin, nend, includeRight=True):
    max = math.pow(2, iend - ibegin)
    result = 0
    for i in range(0, iend - ibegin):
        result += math.pow(2, i) if arr[i] else 0
    if includeRight:
        ret = (nend - nbegin) / (max - 1) * result + nbegin
    else:
        ret = (nend - nbegin) / max * result + nbegin
    return ret


def boolarr2num(arr, ibegin, iend):
    result = 0
    for i in range(0, iend - ibegin):
        result += math.pow(2, i) if arr[i] else 0
    return result


# 正态分布取随机数
def normalvariateIndex(begin, end, pivot, privilegeWeight=2):
    if end == begin:
        return end
    normalPivot = (pivot - begin) / (end - begin)
    ran = random.normalvariate(normalPivot, normalPivot / privilegeWeight)
    while ran > 1 or ran < 0:
        ran = random.normalvariate(normalPivot, normalPivot / privilegeWeight)
    return round(ran * (end - begin + 1) - 0.5)

# def myfitnessFunc(indi):
#     sol = boolarr2double(indi.getGene(0, 59), 0, 60, 1.5, 2.5, True)
#     return -math.fabs(5 * sol * sol * sol - 19 * sol * sol + 18.05 * sol)
#
#
# def getSolution(indi):
#     return boolarr2double(indi.getGene(0, 59), 0, 60, 1.5, 2.5, True)
#
#
# def getResult(indi):
#     sol = getSolution(indi)
#     return 5 * sol * sol * sol - 19 * sol * sol + 18.05 * sol
#
#
# g = GeneticCalculator(50, 50, 60, myfitnessFunc)
# for i in range(500):
#     g.calculate(1)
#     print('第' + str(i + 1) + '代,当前最优根:' + str(getSolution(g.currentBest)) + ',当前最优解:' + str(getResult(g.currentBest)))
