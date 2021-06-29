# importing necessary packages
import numpy as np
import Genetic
from tflearn.data_utils import image_preloader
import random
import tflearn
import tensorflow as ptf
# 训练次数,建议20-30
epoch = 60
# 训练一次分批的大小
batch_size = 4
tf = ptf.compat.v1

# reading images from a text file
ROOT_FOLDER = './dataset/'
IMAGE_FOLDER = ROOT_FOLDER + 'imageData'
TRAIN_DATA = ROOT_FOLDER + 'trainData/train_data.txt'
TEST_DATA = ROOT_FOLDER + 'testData/test_data.txt'
VALIDATION_DATA = ROOT_FOLDER + 'validationData/validation_data.txt'

# -------------------------------------------------------------------------------------------------------------------------

train_proportion = 0.8
test_proportion = 0.1


# -------------------------------------------------------------------------------------------------------------------------

# def create_train_test(train_test_prop, detector_value, pixel, convolution_node):
def create_train_test(indi):
    # detector_value = int(Genetic.boolarr2num(indi.getGene(0,4),0,5)+1)
    # pixel = int(Genetic.boolarr2num(indi.getGene(5,9),0,4)+1)
    # convolution_node = int(Genetic.boolarr2num(indi.getGene(10,19),0,10)+1)
    detector_value = 19
    pixel = 14
    convolution_node = 614

    print('args: ' + str(detector_value) + ' ' + str(pixel) + ' ' + str(convolution_node))

    # 无用参数！
    # 该函数是对每个个体遍历使用的
    # 导入train_data.txt中写的图片文件,规定图片大小为64*64,不符合的大小会被调整.
    # detector_value、pixel、convolution_node都是由当前个体的基因得出
    # 导入后X为图片路径，Y为对应的图片class数值
    X_train, Y_train = image_preloader(TRAIN_DATA, image_shape=(pixel, pixel), mode='file', categorical_labels=True,
                                       normalize=True)
    X_test, Y_test = image_preloader(TEST_DATA, image_shape=(pixel, pixel), mode='file', categorical_labels=True,
                                     normalize=True)

    # 图片输入占位，设定好要输入数据的类型，形状和名称进行占位，此时并不真正输入数据
    # 真正赋值一般用sess.run(feed_dict = {x:xs, y_:ys})，其中x,y_是用placeholder创建出来的
    x = tf.placeholder(tf.float32, shape=[None, pixel, pixel, 3], name='input_image')
    # 分类输入占位
    # y_ = tf.placeholder(tf.float32, shape=[None, 4], name='input_class')
    y_ = tf.placeholder(tf.float32, shape=[None, 3], name='input_class')

    # 输入层
    # reshaping input for convolutional operation in tensorflow
    # '-1' states that there is no fixed batch dimension, 28x28(=784) is reshaped from 784 pixels and '1' for a single
    # channel, i.e a gray scale image

    x_input = x
    # first convolutional layer with 32 output filters, filter size 5x5, stride of 2,same padding, and RELU activation.
    # I am not adding bias, but one could add bias.Optionally you can add max pooling layer as well

    # 第二个基因决定卷积核的数量，对应单通道特征图数？卷积核5*5，步长1,1,1,1
    conv_layer1 = tflearn.layers.conv.conv_2d(x_input, nb_filter=detector_value, filter_size=5, strides=[1, 1, 1, 1],
                                              padding='same', activation='relu', regularizer="L2", name='conv_layer_1')

    # 2x2 max pooling layer

    out_layer1 = tflearn.layers.conv.max_pool_2d(conv_layer1, 2)

    # second convolutional layer
    conv_layer2 = tflearn.layers.conv.conv_2d(out_layer1, nb_filter=detector_value, filter_size=5, strides=[1, 1, 1, 1],
                                              padding='same', activation='relu', regularizer="L2", name='conv_layer_2')
    out_layer2 = tflearn.layers.conv.max_pool_2d(conv_layer2, 2)
    # fully connected layer

    fcl = tflearn.layers.core.fully_connected(out_layer2, convolution_node, activation='relu')

    # 80%的神经元抑制概率
    fcl_dropout = tflearn.layers.core.dropout(fcl, 0.8)
    # y_predicted = tflearn.layers.core.fully_connected(fcl_dropout, 4, activation='softmax', name='output')

    y_predicted = tflearn.layers.core.fully_connected(fcl_dropout, 3, activation='softmax', name='output')
    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted + np.exp(-10)), reduction_indices=[1]))
    # optimiser -
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # calculating accuracy of our model
    correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # session parameters
    sess = tf.InteractiveSession()
    # initialising variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # grabbing the default graph
    g = tf.get_default_graph()

    # every operations in our graph
    [op.name for op in g.get_operations()]

    # epoch = 1  # run for more iterations according your hardware's power
    # # change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
    # batch_size = 1
    no_itr_per_epoch = len(X_train) // batch_size

    no_itr_per_epoch
    n_test = len(X_test)  # number of test samples

    for iteration in range(epoch):

        previous_batch = 0
        # Do our mini batches:
        for i in range(no_itr_per_epoch):
            current_batch = previous_batch + batch_size
            x_input = X_train[previous_batch:current_batch]
            x_images = np.reshape(x_input, [batch_size, pixel, pixel, 3])

            y_input = Y_train[previous_batch:current_batch]
            # y_label = np.reshape(y_input, [batch_size, 4])
            y_label = np.reshape(y_input, [batch_size, 3])
            previous_batch = previous_batch + batch_size

            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: x_images, y_: y_label})

        # t = X_test[0:n_test]
        # print(str(len(t))+' '+str(len(t[0]))+' '+str(len(t[0][0]))+' '+str(len(t[0][0][0]))+' '+str(t[0][0][0][0]))

        x_test_images = np.reshape(X_test[0:n_test], [n_test, pixel, pixel, 3])
        # y_test_labels = np.reshape(Y_test[0:n_test], [n_test, 4])
        y_test_labels = np.reshape(Y_test[0:n_test], [n_test, 3])
        Accuracy_test = sess.run(accuracy,
                                 feed_dict={
                                     x: x_test_images,
                                     y_: y_test_labels
                                 })
        Accuracy_test = round(Accuracy_test * 100, 3)

        print("Accuracy ::  Test_set {} % " .format(Accuracy_test))
    return Accuracy_test


# -------------------------------------------------------------------------------------------------------------------------

def cal_pop_fitness(new_population):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    # 计算每个个体的适应性
    # 个体适应性函数是当前个体的每项评估指标的加权求和？
    fitness = []
    # 对每个个体计算适应性，添加到fitness数组中，然后返回
    for i in new_population:
        fitness.append(create_train_test(i[0], i[1], i[2], i[3]))
    return np.array(fitness)


# -------------------------------------------------------------------------------------------------------------------------
'''
def select_mating_pool(pop, fitness, num_parents):
        parents[parent_num, :] = pop[max_fitness_idx, :]
'''


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    # print(pop)
    parents = np.empty((num_parents, len(pop[0])))

    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))

        max_fitness_idx = max_fitness_idx[0][0]

        parents[parent_num, :] = pop[max_fitness_idx, :]
        # parents.append(pop[max_fitness_idx].copy())

        fitness[max_fitness_idx] = -99

    return parents


# -------------------------------------------------------------------------------------------------------------------------

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring wi ll have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


# -------------------------------------------------------------------------------------------------------------------------
def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.

    index = random.randint(1, 3)

    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.

        random_value = random.randint(-1.0, 1.0)

        offspring_crossover[idx, index] = offspring_crossover[idx, index] * (2 ** random_value)
        # It will multyply with 2 or devide by 2

    return offspring_crossover
# ----------------------------------------------------------------------------------------------------------------------------
