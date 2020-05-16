#!/usr/bin/env python3
"""
Evaluate
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """function that evaluates the output of a nn"""
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(save_path + '.meta')
        loader.restore(sess, save_path)

        # the following approach will NOT work here due to
        # "NameError: name 'x' is not defined" -> globals() call instead
        # params = {'x': x, 'y': y, 'y_pred': y_pred, 'accuracy': accuracy,
        #           'loss': loss, 'train_op': train_op}
        # for k, v in params.items():
        #     v = tf.get_collection(k)[0]

        # globals() creates variables with names given as strings at run-time
        var_names = ['x', 'y', 'y_pred', 'accuracy', 'loss']
        for var_name in var_names:
            globals()[var_name] = tf.get_collection(var_name)[0]
            # print(globals()[var_name])

        y_pred = sess.run(globals()['y_pred'], feed_dict={x: X, y: Y})
        loss = sess.run(globals()['loss'], feed_dict={x: X, y: Y})
        acc = sess.run(globals()['accuracy'], feed_dict={x: X, y: Y})
        # print(y_pred)
        # print(acc)
        # print(loss)
    return y_pred, acc, loss
