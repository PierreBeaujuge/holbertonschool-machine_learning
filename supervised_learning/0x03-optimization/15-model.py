#!/usr/bin/env python3
"""
Assemble Model
"""
import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """function that builds, trains, and saves a nn model in tensorflow"""

    # create placeholders for input data and labels
    nx = Data_train[0].shape[1]
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    classes = Data_train[1].shape[1]
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    # build forward propagation graph
    # include batch normalization filters
    for i in range(len(layers)):
        if i == 0:
            y_pred = x
        initializer = tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG")
        if i == len(layers) - 1:
            layer = tf.layers.Dense(units=layers[i],
                                    activation=None,
                                    kernel_initializer=initializer,
                                    name='layer')
            if len(activations) and activations[i]:
                y_pred = activations[i](layer(y_pred))
            else:
                y_pred = layer(y_pred)
            break
        layer = tf.layers.Dense(units=layers[i],
                                activation=None,
                                kernel_initializer=initializer,
                                name='layer')
        # Important: make a copy of layer(y_pred) here
        # before the batch normalization operation
        Z = layer(y_pred)
        m, v = tf.nn.moments(Z, axes=[0])
        beta = tf.Variable(
            tf.zeros(shape=(1, layers[i]), dtype=tf.float32),
            trainable=True, name='beta'
        )
        gamma = tf.Variable(
            tf.ones(shape=(1, layers[i]), dtype=tf.float32),
            trainable=True, name='gamma'
        )
        Z_b_norm = tf.nn.batch_normalization(
            x=Z, mean=m, variance=v, offset=beta, scale=gamma,
            variance_epsilon=epsilon, name=None
        )
        if len(activations) and activations[i]:
            y_pred = activations[i](Z_b_norm)
        else:
            y_pred = Z_b_norm

    # define graph operation for accuracy
    label = tf.argmax(y, axis=1)
    pred = tf.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))

    # define graph operation for loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    # define graph operation for the train method
    # update the learning rate using inverse time decay
    global_step = tf.Variable(0, trainable=False)
    alpha = tf.train.inverse_time_decay(
        learning_rate=alpha, global_step=global_step, decay_steps=1,
        decay_rate=decay_rate, staircase=True, name=None
    )
    train_op = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon,
        use_locking=False, name='Adam'
    ).minimize(loss)
    # note: passing global_step to minimize would
    # increment global_step by one on every gradient
    # descent as opposed to every epoch
    # ).minimize(loss, global_step=global_step)

    # add variables and operations to collections
    params = {'x': x, 'y': y, 'y_pred': y_pred, 'accuracy': accuracy,
              'loss': loss, 'train_op': train_op}
    for k, v in params.items():
        tf.add_to_collection(k, v)

    # call to global init, open session
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):

            loss_t = sess.run(loss,
                              feed_dict={x: Data_train[0], y: Data_train[1]})
            acc_t = sess.run(accuracy,
                             feed_dict={x: Data_train[0], y: Data_train[1]})
            loss_v = sess.run(loss,
                              feed_dict={x: Data_valid[0], y: Data_valid[1]})
            acc_v = sess.run(accuracy,
                             feed_dict={x: Data_valid[0], y: Data_valid[1]})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(loss_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            if epoch < epochs:

                # shuffle input data before each new epoch
                shuffle = np.random.permutation(Data_train[0].shape[0])
                X_shuff = Data_train[0][shuffle]
                Y_shuff = Data_train[1][shuffle]

                # define iteration range for the mini-batch training
                batches_float = Data_train[0].shape[0] / batch_size
                batches_int = int(Data_train[0].shape[0] / batch_size)

                # gradient descent step
                # reinitialized to 0 between epochs
                step = 0

                # initialize and update the learning rate alpha
                gs = sess.run(global_step.assign(epoch))
                a = sess.run(alpha)

                for i in range(0, batches_int + 1):
                    step += 1

                    # Important: make copies of X_shuff and Y_shuff
                    if i == batches_int:
                        if batches_float > batches_int:
                            X_batch = X_shuff[i * batch_size:]
                            Y_batch = Y_shuff[i * batch_size:]
                        else:
                            break
                    else:
                        X_batch = X_shuff[i * batch_size: (i + 1) * batch_size]
                        Y_batch = Y_shuff[i * batch_size: (i + 1) * batch_size]

                    # then pass the copies on to feed_dict / train_op
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    # print after every 100 gradient descent steps
                    if step % 100 == 0:
                        loss_b = sess.run(loss, feed_dict={
                            x: X_batch, y: Y_batch})
                        acc_b = sess.run(accuracy, feed_dict={
                            x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(loss_b))
                        print("\t\tAccuracy: {}".format(acc_b))

        save_path = saver.save(sess, save_path)
    return save_path
