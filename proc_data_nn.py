import sklearn as skl
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    dataset = pd.read_csv('data.csv', usecols=['combined_shot_type', 'seconds_remaining', 'shot_distance',
                                               'playoffs', 'shot_zone_range', 'shot_zone_area', 'shot_made_flag'])
    return dataset


def convert2onehot(d2onehot):
    # covert data to onehot representation
    return pd.get_dummies(d2onehot)


# data_df = load_data().dropna()
# data_df['shot_fail_flag'] = 1-data_df.loc[:, 'shot_made_flag']
# # print(data_df.head(5))
# data = convert2onehot(data_df)
# print('Onehot data:\n', data[:5])
# print("Num of data:\n ", data.shape, "\n")
# for name in data_df.keys():
#     print(name, pd.unique(data_df[name]))

# Preprocessing & separate training sets
x_ = np.load('../data/soil_pca_plant_3600x88.npy')
y_ = np.load('../data/label.npy')[:3600]
y_ = convert2onehot(y_)
print('x:', x_.shape)
print('y:', y_.shape)

shuffle = np.hstack((y_, x_)).copy()
# print('before shuffle:', b4shuffle)
np.random.shuffle(shuffle)  # shuffle
# print('after shuffle:', b4shuffle.shape)
# print(b4shuffle)
# 4: Separate
sep = int(0.3 * len(x_))
train_x = shuffle[:sep, 4:].copy()
train_y = shuffle[:sep, 0:4].copy()
test_x = shuffle[sep:, 4:].copy()
test_y = shuffle[sep:, 0:4].copy()

# x_combined = np.vstack((train_x, test_x))
# y_combined = np.vstack((train_y, test_y))
# print(train_x[:15])
# print(train_y[:15])


# build network
with tf.variable_scope('Inputs'):
    tfx = tf.placeholder(tf.float32, [None, 82], 'Input_x')
    tfy = tf.placeholder(tf.float32, [None, 4], 'Input_y')

print('tfx shape', tfx.shape)

with tf.variable_scope('Layers'):
    l1 = tf.layers.dense(tfx, 256, tf.nn.relu, name="L1")  # 全连接层
    l2 = tf.layers.dense(tfx, 256, tf.nn.tanh, name="L2")
    # l3 = tf.layers.dense(l2, 128, tf.nn.relu, name="L3")
    # l4 = tf.layers.dense(l3, 64, tf.nn.relu, name="L4")
    out = tf.layers.dense(l2, 4, tf.nn.relu, name="L5")

prediction = tf.nn.softmax(out, name="Prediction")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out, scope='loss')
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
optima = tf.train.AdadeltaOptimizer(learning_rate=0.03)
train_optima = optima.minimize(loss)
tf.summary.scalar('Loss', loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer()))  # accuracy is in local variable init
writer = tf.summary.FileWriter("/Users/Epilo/Documents/CodeTest/py_BigData", sess.graph)

# training
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
accuracies, steps = [], []
for t in range(10000):
    # training
    sess.run(train_optima, feed_dict={tfx: train_x, tfy: train_y})

    if t % 50 == 0:
        # testing
        acc_, pred_, loss_ = sess.run([accuracy, prediction, loss], {tfx: test_x, tfy: test_y})
        accuracies.append(acc_)
        steps.append(t)
        # print(acc_.__class__)
        print("Step: %i" % t, "| ACC: %.6f" % acc_, "| Loss: %.6f" % (loss_*10000))

        # visualize testing
        ax1.cla()
        for c in range(4):
            bp = ax1.bar(c+0.1, height=sum((np.argmax(pred_, axis=1) == c+1)),
                         width=0.2, color='red')
            bt = ax1.bar(c-0.1, height=sum((np.argmax(test_y, axis=1) == c+1)),
                         width=0.2, color='blue')
        ax1.set_xticks(range(2), ['success', 'failure'])
        ax1.legend(handles=[bp, bt], labels=["prediction", "target"])
        ax1.set_ylim((0, 1000))
        ax2.cla()
        ax2.plot(steps, accuracies, label="accuracy")
        ax2.set_ylim(ymax=1)
        ax2.set_ylabel("accuracy")
        plt.pause(0.01)

plt.ioff()
plt.show()

print(1)
