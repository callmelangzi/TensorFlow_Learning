```python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r'D:\迅雷下载\mnist', one_hot=True)

train_data = mnist.train.images
train_label = mnist.train.labels
test_data = mnist.test.images
test_label = mnist.test.labels
vaild_data = mnist.validation.images
vaild_label = mnist.validation.labels

# img = train_data[10].reshape(28,28)
# plt.imshow(img)
# plt.show()

input = 784
out_put = 10
layer_node = 500
batch_size = 200
learning_rate_base = 0.1
learning_rate_decay = 0.99
regularizer = 0.0001
epochs = 50000
moving_average_decay = 0.99



def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None :
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1 = get_weight([input,layer_node],regularizer)
    b1 = get_bias([layer_node])
    y1 = tf.nn.relu(tf.matmul(x,w1) + b1)

    w2 = get_weight([layer_node,out_put],regularizer)
    b2 = get_bias([out_put])
    y = tf.matmul(y1,w2) + b2

    return y


def bpnn(mnist):
    x = tf.placeholder(tf.float32,[None,input])
    y_ = tf.placeholder(tf.float32,[None,out_put])
    y = forward(x,regularizer)
    loss_list = []
    global_step = tf.Variable(0,trainable=False) # 当前轮数计数器设定为不可训练类型
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step,
        mnist.train.num_examples / batch_size,
        learning_rate_decay,
        staircase=True
    )
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1,epochs):
            xs, ys = mnist.train.next_batch(batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if epoch % 1000 == 0:
                print('epoch is ', epoch, 'loss is ', loss_value)

            if epoch % 500 == 0:
                loss_list.append(loss_value)
    plt.plot(loss_list)
    plt.title('the loss solution')
    plt.show()

bpnn(mnist)

```
