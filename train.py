import tensorflow as tf
import numpy as np
import time
import data
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

# %matplotlib inline
tf.set_random_seed(55)
np.random.seed(55)


class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer,
                 f_bias=1.0, L2=False, h_act=tf.tanh,
                 init_h=None, init_c=None):
        # 属性
        self.incoming = incoming  # 输入数据
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer  # 初始化方法
        self.f_bias = f_bias  # 遗忘门的初始偏移量
        self.h_act = h_act  # 这里可以选择LSTM的hidden state的激活函数
        self.type = 'lstm'  # 区分gru
        # 如果没有提供最初的hidden state和memory cell，会全部初始为0
        if init_h is None and init_c is None:
            # If init_h and init_c are not provided, initialize them
            # the shape of init_h and init_c is [n_samples, D_cell]
            self.init_h = tf.matmul(self.incoming[0, :, :], tf.zeros([self.D_input, self.D_cell]))
            self.init_c = self.init_h
            self.previous = tf.stack([self.init_h, self.init_c])
        # LSTM所有需要学习的参数
        # 每个都是[W_x, W_h, b_f]的tuple
        self.igate = self.Gate()
        self.fgate = self.Gate(bias=f_bias)
        self.ogate = self.Gate()
        self.cell = self.Gate()
        # 因为所有的gate都会乘以当前的输入和上一时刻的hidden state
        # 将矩阵concat在一起，计算后再逐一分离，加快运行速度
        # W_x的形状是[D_input, 4*D_cell]
        self.W_x = tf.concat(values=[self.igate[0], self.fgate[0], self.ogate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.igate[1], self.fgate[1], self.ogate[1], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.igate[2], self.fgate[2], self.ogate[2], self.cell[2]], axis=0)
        # 对LSTM的权重进行L2 regularization
        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    # 初始化gate的函数
    def Gate(self, bias=0.001):
        # Since we will use gate multiple times, let's code a class for reusing
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    # 大矩阵乘法运算完毕后，方便用于分离各个gate
    def Slice_W(self, x, n):
        # split W's after computing
        return x[:, n * self.D_cell:(n + 1) * self.D_cell]
        # 每个time step需要运行的步骤

    def Step(self, previous_h_c_tuple, current_x):
        # 分离上一时刻的hidden state和memory cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 统一在concat成的大矩阵中一次完成所有的gates计算
        gates = tf.matmul(current_x, self.W_x) + tf.matmul(prev_h, self.W_h) + self.b
        # 分离输入门
        i = tf.sigmoid(self.Slice_W(gates, 0))
        # 分离遗忘门
        f = tf.sigmoid(self.Slice_W(gates, 1))
        # 分离输出门
        o = tf.sigmoid(self.Slice_W(gates, 2))
        # 分离新的更新信息
        c = tf.tanh(self.Slice_W(gates, 3))
        # 利用gates进行当前memory cell的计算
        current_c = f * prev_c + i * c
        # 利用gates进行当前hidden state的计算
        current_h = o * self.h_act(current_c)
        return tf.stack([current_h, current_c])


class GRUcell(object):
    """
    create parameters and step function
    """

    def __init__(self, incoming, D_input, D_cell, initializer, L2=False, init_h=None):

        # var
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer
        self.type = 'gru'

        if init_h is None:
            # If init_h is not provided, initialize it
            # the shape of init_h is [n_samples, D_cell]
            self.init_h = tf.matmul(self.incoming[0, :, :], tf.zeros([self.D_input, self.D_cell]))
            self.previous = self.init_h

        # parameters, each of which has W_x W_h b_f
        self.rgate = self.Gate()
        self.ugate = self.Gate()
        self.cell = self.Gate()

        # to speed up computation. W_x has the shape of [D_input, 3*D_cell]
        self.W_x = tf.concat(values=[self.rgate[0], self.ugate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.rgate[1], self.ugate[1], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.rgate[2], self.ugate[2], self.cell[2]], axis=0)

        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    def Gate(self, bias=0.001):
        # Since we will use gate multiple times, let's code a class for reusing
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    def Slice_W(self, x, n):
        # split W's after computing
        return x[:, n * self.D_cell:(n + 1) * self.D_cell]

    def Step(self, prev_h, current_x):

        Wx = tf.matmul(current_x, self.W_x) + self.b
        Wh = tf.matmul(prev_h, self.W_h)

        r = tf.sigmoid(self.Slice_W(Wx, 0) + self.Slice_W(Wh, 0))
        u = tf.sigmoid(self.Slice_W(Wx, 1) + self.Slice_W(Wh, 1))

        c = tf.tanh(self.Slice_W(Wx, 2) + r * self.Slice_W(Wh, 2))

        current_h = (1 - u) * prev_h + u * c

        return current_h


def RNN(cell, cell_b=None, merge='sum'):
    """
    Note that the input shape should be [n_steps, n_sample, D_output],
    and the output shape will also be [n_steps, n_sample, D_output].
    If the original data has a shape of [n_sample, n_steps, D_input],
    use 'inputs_T = tf.transpose(inputs, perm=[1,0,2])'.
    """

    # forward rnn loop
    hstates = tf.scan(fn=cell.Step,
                      elems=cell.incoming,
                      initializer=cell.previous,
                      name='hstates')
    if cell.type == 'lstm':
        hstates = hstates[:, 0, :, :]
    # reverse the input sequence
    if cell_b is not None:
        incoming_b = tf.reverse(cell.incoming, axis=[0])

        # backward rnn loop
        b_hstates_rev = tf.scan(fn=cell_b.Step,
                                elems=incoming_b,
                                initializer=cell_b.previous,
                                name='b_hstates')
        if cell_b.type == 'lstm':
            b_hstates_rev = b_hstates_rev[:, 0, :, :]

        b_hstates = tf.reverse(b_hstates_rev, axis=[0])

        if merge == 'sum':
            hstates = hstates + b_hstates
        else:
            hstates = tf.concat(values=[hstates, b_hstates], axis=2)
    return hstates


# def attention(inputs, target, l2_reg_lambda=0):
#     # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
#     if isinstance(inputs, tuple):
#         inputs_ = tf.concat(inputs, 2)
#     else:
#         inputs_ = inputs
#     batch_size = inputs_.get_shape()[0].value
#     sequence_length = inputs_.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
#     word_dim = inputs_.get_shape()[2].value  # hidden size of the RNN layer
#
#     W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([word_dim, int(word_dim / 4)], stddev=0.1))
#     ht_W = tf.reshape(tf.matmul(tf.reshape(inputs_, [-1, word_dim]), W_omega), [-1, sequence_length, int(word_dim / 4)])
#     ht_W_tar = tf.matmul(ht_W, target)
#     at = tf.nn.softmax(ht_W_tar)
#     output = tf.reduce_sum(tf.multiply(inputs_, at), reduction_indices=1)
#
#     return output


def attention(inputs, attention_size, l2_reg_lambda=0):
    """
    Attention mechanism layer.
    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param attention_size: linear size of attention weights
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

    W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    # if l2_reg_lambda > 0:
    #    l2_loss += tf.nn.l2_loss(W_omega)
    #    l2_loss += tf.nn.l2_loss(b_omega)
    #    l2_loss += tf.nn.l2_loss(u_omega)
    #    tf.add_to_collection('losses', l2_loss)

    return output


def weight_init(shape):
    initial = tf.random_uniform(shape, minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
    return tf.Variable(initial, trainable=True)


def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)


def orthogonal_initializer(shape, scale=1.0):
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[0], :shape[1]], dtype=tf.float32, trainable=True)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, trainable=True)


def shufflelists(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def Standardize(seq):
    # subtract mean
    centerized = seq - np.mean(seq, axis=0)
    # divide standard deviation
    normalized = centerized / np.std(centerized, axis=0)
    return normalized


train = data.Data(["data/data_{}-1-1_{}-1-1.csv".format(i, i + 1) for i in range(2009, 2016)])
valid = data.Data(["data/data_2016-1-1_2017-1-1.csv"])
test = data.Data(["data/data_2017-1-1_2018-1-1.csv"])
print('num of train sequences:%s' % len(train.data))
print('num of valid sequences:%s' % len(valid.data))
print('num of test sequences:%s' % len(test.data))

D_input = 6
D_label = 20
learning_rate = 1e-4
L2_penalty = 5e-3
mini_batch = 100
sequence_length = 7 * 240
attention_size = 100
num_units = 100
RNN_cell = "LSTM"

inputs = tf.placeholder(tf.float32, [None, sequence_length, D_input], name="inputs")
labels = tf.placeholder(tf.float32, [None, D_label], name="labels")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
input = tf.transpose(inputs, perm=[1, 0, 2])

if RNN_cell == "GRU":
    rnn_cell = GRUcell(input, D_input, num_units, orthogonal_initializer)
elif RNN_cell == "LSTM":
    rnn_cell = LSTMcell(input, D_input, num_units, orthogonal_initializer)
rnn = RNN(rnn_cell)

rnn_ = tf.transpose(rnn, perm=[1, 0, 2])  # turn into (batch_size, steps, rnn_size*2)
a_re_ = attention(rnn_, attention_size)

W2 = weight_init([num_units, D_label])
b2 = bias_init([D_label])
re2 = tf.matmul(a_re_, W2) + b2
output = tf.nn.softmax(re2)

loss = tf.reduce_mean(tf.square((output - labels)))
tf.summary.scalar('loss', loss)

L2_total = tf.nn.l2_loss(W2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss + L2_penalty * L2_total)

y_pred = tf.argmax(output, 1)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

merged = tf.summary.merge_all()
test_writer = tf.summary.FileWriter(str(time.strftime("%Y-%m-%d")) + "/test")
train_writer = tf.summary.FileWriter(str(time.strftime("%Y-%m-%d")) + "/train")

tf.global_variables_initializer().run()


def train_epoch(EPOCH):
    for k in range(EPOCH):
        x, y = train.next_batch(100)
        sess.run(train_step, feed_dict={inputs: x, labels: y, keep_prob: 0.3})
        if k % 10 == 0:
            def cal_acc_f1(x, y):
                acc = sess.run(accuracy, feed_dict={inputs: x, labels: y, keep_prob: 1})
                y_t = [list(i).index(1) for i in y]
                y_p = list(sess.run(y_pred, feed_dict={inputs: x, labels: y, keep_prob: 1}))
                f1 = f1_score(y_true=y_t, y_pred=y_p, average='macro')
                return round(acc, 3), round(f1, 3)

            train_batch = train.next_batch(500)
            test_batch = test.next_batch(500)
            train_acc, train_f1 = cal_acc_f1(train_batch[0], train_batch[1])
            test_acc, test_f1 = cal_acc_f1(test_batch[0], test_batch[1])
            print("EPOCH:{}\ttrain accuracy:{}\ttrain macro-F1:{}".format(k, train_acc, train_f1, 3))
            print("\t\ttest accuracy:{}\ttest macro-F1:{}".format(test_acc, test_f1))
            print()
            train_result = sess.run(merged, feed_dict={inputs: train_batch[0], labels: train_batch[1], keep_prob: 1})
            train_writer.add_summary(train_result, k)
            test_result = sess.run(merged, feed_dict={inputs: test_batch[0], labels: test_batch[1], keep_prob: 1})
            test_writer.add_summary(test_result, k)
        if k % 100 == 0:
            saver = tf.train.Saver()
            model_path = "E:\Code\Python\DataMining\model\model.ckpt"
            save_path = saver.save(sess, model_path)
            print("Model saved. {}".format(save_path))
            print()


def predict(data):
    saver = tf.train.Saver()
    saver.restore(sess, "E:\Code\Python\DataMining\model\model.ckpt")
    result = sess.run(y_pred, feed_dict={inputs: data, keep_prob: 1})
    return result


def demonstrate():
    plt.axis([0, 1000, 0, 1])
    plt.ion()

    while True:
        import random
        day = random.randint(0, train.days - 8)
        x_, x = train.gen_batch_by_day(day)
        x_ = x_[:1680].reshape([1, 1680, 6])
        x = [i[0] for i in x]

        output = predict(x_[:240 * 7])[0]
        co = output / 100 + 0.9
        distribution = (x[-1] * co, x[-1] * (co + 0.01))

        days = [i for i in range(8 * 240)]
        price = [i for i in x]
        pred_low = [None for i in range(1680)] + [distribution[0]] * 240
        pred_high = [None for i in range(1680)] + [distribution[1]] * 240
        print(price[1680], pred_high[-1], pred_low[-1])
        plt.cla()
        # print(price[(day + 7) * 240])
        plt.plot(days, price)
        plt.plot(days, pred_low)
        plt.plot(days, pred_high)
        plt.pause(1)


# def demonstrate1():
#     plt.axis([0, 1000, 0, 1])
#     plt.ion()
#
#     pred = [None for i in range(1680)]
#     price = []
#     for day in range(valid.days):
#         x_, x, co = valid.gen_batch_by_day(day)
#         x_ = x_[:1680].reshape([1, 1680, 6])
#         x = [i[0] for i in x]
#
#         prediction = predict(x_[:240 * 7]) * co / 0.6
#
#         days = [i for i in range((day + 8) * 240)]
#         if price != []:
#             price = price[:day * 240] + [i for i in x]
#         else:
#             price = [i for i in x]
#         pred += [prediction] * 240
#         plt.cla()
#         # print(price[(day + 7) * 240])
#         plt.plot(days, price)
#         plt.plot(days, pred)
#         plt.pause(0.0033)


if __name__ == "__main__":
    t0 = time.time()
    train_epoch(100000)
    t1 = time.time()
    print(" %f seconds" % round((t1 - t0), 2))
    # demonstrate()
