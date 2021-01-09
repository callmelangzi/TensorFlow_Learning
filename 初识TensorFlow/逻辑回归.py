import tensorflow as tf
from sklearn.datasets import make_moons
import numpy as np
from sklearn.model_selection import train_test_split


class Ts_Logist_Regression:
    def prepare(self, data_size,test_size):
        X_moons, y_moons = make_moons(data_size, noise=0.1, random_state=42)
        X_moons_add_bias = np.c_[np.ones((data_size, 1)), X_moons]
        y_moons_column = y_moons.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_moons_add_bias, y_moons_column,test_size=test_size)
        return X_train, X_test, y_train, y_test

    def rnd_batch(self,X_train,y_train,batch_size):
        rnd = np.random.randint(0,len(X_train),batch_size)
        X_batch = X_train[rnd]
        y_batch = y_train[rnd]
        return X_batch,y_batch

    def logist_train(self, X_train, y_train, X_test,y_test,learning_rate,epochs, batch_size):
        m,n = X_train.shape
        X = tf.placeholder(dtype=tf.float32, shape=(None,n),name='X')
        y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='y')
        theta = tf.Variable(tf.random_uniform([n,1],-1,1,seed=42), name='theta')
        tmp1 = tf.matmul(X,theta)
        y_proba = tf.sigmoid(tmp1)
        loss = tf.losses.log_loss(tf.sigmoid(tmp1), y)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(1,epochs):
                for batch_index in range(batch_size):
                    X_batch,y_batch = self.rnd_batch(X_train,y_train,batch_size)
                    sess.run(train, feed_dict={X:X_batch,y:y_batch})

                loss_val = sess.run(loss,feed_dict={X:X_train,y:y_train})

                if epoch % 100 == 0:
                    print(loss_val)

            y_proba_val = y_proba.eval(feed_dict={X: X_test})

        return y_proba_val, y_test

    def result(self,y_prob,y_test ):

        y_prob = (y_prob >= 0.5)
        y_test = (y_test >= 0.5)
        return sum(y_prob == y_test) / len(y_test)



if __name__ == '__main__':
    data_szie = 1000
    test_size = 0.3
    learning_rate = 0.01
    epochs = 1000
    batch_size= 100
    s = Ts_Logist_Regression()
    X_train, X_test, y_train, y_test = s.prepare(data_szie,test_size)
    y_proba_val, y_test = s.logist_train(X_train,y_train,X_test,y_test,learning_rate,epochs,batch_size)
    print(s.result(y_proba_val, y_test))
