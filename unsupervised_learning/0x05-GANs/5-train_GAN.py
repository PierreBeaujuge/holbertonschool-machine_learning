#!/usr/bin/env python3
"""
5-train_GAN.py
"""
import tensorflow as tf
import numpy as np
train_discriminator = __import__('2-train_discriminator').train_discriminator
train_generator = __import__('3-train_generator').train_generator
sample_Z = __import__('4-sample_Z').sample_Z
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
generator = __import__('0-generator').generator


def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'):
    """function that trains a GAN"""

    # Visualize X
    # print("X:", X)
    # print("X.shape:", X.shape)

    # Input images for the discriminator
    x = tf.placeholder(tf.float32, shape=(None, X.shape[1]), name='X')
    # Input noise for the generator
    z = tf.placeholder(tf.float32, shape=(None, Z_dim), name='Z')

    # Define the size of the dataset
    buffer_size = X.shape[0]
    # print("buffer_size:", buffer_size)

    # Instantiate the train_ops for the discriminator and the generator
    D_loss, D_train_op = train_discriminator(z, x)
    G_loss, G_train_op = train_generator(z)

    # Add params to graph collection
    params = {'x': x, 'z': z, 'D_loss': D_loss, 'G_loss': G_loss,
              'D_train_op': D_train_op, 'G_train_op': G_train_op}
    for k, v in params.items():
        tf.add_to_collection(k, v)

    # Instantiate a saver and start a session
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Create folder to save images (.png)
        # if not os.path.exists('saved_images/'):
        #     os.makedirs('saved_images/')
        # j = 0

        # Iterate over epochs
        for i in range(epochs + 1):

            # Call to next_batch(), function that batches/shuffles
            Xb = next_batch(batch_size, X)

            # Create input for the generator
            Zb = sample_Z(batch_size, Z_dim)

            # Run discriminator train_op
            _, D_loss_t = sess.run([D_train_op, D_loss],
                                   feed_dict={x: Xb,
                                              z: Zb})
            # Run generator train_op
            _, G_loss_t = sess.run([G_train_op, G_loss],
                                   feed_dict={z: Zb})

            if i == 0 or i % 1000 == 0 or i == epochs:
                print('Epoch: {}'.format(i))
                print('D_loss: {}'.format(D_loss_t))
                print('G_loss: {}'.format(G_loss_t))

            # if i == epochs:
                samples = sess.run(generator(z), feed_dict={z: Zb})
                # Call to plot()
                fig = plot(samples)
                # plt.savefig('saved_images/{}.png'.format(str(i).zfill(3)),
                #             bbox_inches='tight')
                # j += 1
                plt.close(fig)

        save_path = saver.save(sess, save_path)
    return save_path

def next_batch(batch_size, X):
    """function that returns a batch of 'batch_size' random samples"""
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    X_shuffled = [X[i] for i in idx]

    return np.asarray(X_shuffled)

def plot(samples):
    """function that plots a grid of digit images"""
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[:16]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28))
        # plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.show()

    return fig
