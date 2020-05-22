#!/usr/bin/env python3
"""
Mini-Batch
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """function that trains a nn model using mini-batch gradient descent"""
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        var_names = ['x', 'y', 'accuracy', 'loss', 'train_op', 'y_pred']
        for var_name in var_names:
            globals()[var_name] = tf.get_collection(var_name)[0]

        for epoch in range(epochs + 1):

            loss_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_v = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_v = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(loss_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            if epoch < epochs:
                X_train, Y_train = shuffle_data(X_train, Y_train)
                batches_float = X_train.shape[0] / batch_size
                batches_int = int(X_train.shape[0] / batch_size)
                step = 0

                for i in range(0, batches_int):
                    step += 1
                    sess.run(train_op, feed_dict={
                        x: X_train[i * batch_size: (i + 1) * batch_size],
                        y: Y_train[i * batch_size: (i + 1) * batch_size]
                    })
                    if step % 100 == 0:
                        loss_b = sess.run(loss, feed_dict={
                            x: X_train[i * batch_size: (i + 1) * batch_size],
                            y: Y_train[i * batch_size: (i + 1) * batch_size]
                        })
                        acc_b = sess.run(accuracy, feed_dict={
                            x: X_train[i * batch_size: (i + 1) * batch_size],
                            y: Y_train[i * batch_size: (i + 1) * batch_size]
                        })
                        print("Step {}:".format(step))
                        print("\tCost: {}".format(loss_b))
                        print("\tAccuracy: {}".format(acc_b))
                if batches_float > batches_int:
                    sess.run(train_op, feed_dict={
                        x: X_train[(i + 1) * batch_size:],
                        y: Y_train[(i + 1) * batch_size:]
                    })

        save_path = loader.save(sess, save_path)
    return save_path
