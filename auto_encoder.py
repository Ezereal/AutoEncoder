import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os, sys, cv2
from dataset import *
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Hyper Parameters
weight_decay = 1e-5
batch_size = 128
epoches = 10
channel = 16
model_path = '/disk2/zhaojiacheng/UrbanRegionCls/auto_encoder/'

def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def AutoEncoder(train_x, training):
    with tf.variable_scope('AutoEncoder'):
        # encoder
        conv0 = tf.nn.relu(batch_normalization(tf.layers.conv2d(train_x, use_bias=False, filters=channel*2, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn0'))
        conv1 = tf.nn.relu(batch_normalization(tf.layers.conv2d(conv0, use_bias=False, filters=channel*2, \
                kernel_size=[3, 3], strides=1, padding='SAME'), training=training, scope='bn1'))
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=2, padding='SAME')
        conv2 = tf.nn.relu(batch_normalization(tf.layers.conv2d(conv1, use_bias=False, filters=channel*4, \
                kernel_size=[3, 3], strides=1, padding='SAME'), training=training, scope='bn2'))
        #conv2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=2, padding='SAME')
    
        feature_down = tf.reshape(conv2, [-1, 7*7*64])
        feature_down = tf.nn.relu(batch_normalization(tf.layers.dense(feature_down, 256), \
                training=training, scope='feature_down_bn'))
        feature_out = tf.layers.dense(feature_down, 64)

        feature_up = tf.nn.relu(tf.layers.dense(feature_out, 256))
        feature_up = tf.nn.relu(batch_normalization(tf.layers.dense(feature_up, 7*7*64), \
                training=training, scope='feature_up_bn'))
        feature_up = tf.reshape(feature_up, [-1, 7, 7, 64])

        # decoder
        #deconv4 = tf.nn.relu(batch_normalization(tf.layers.conv2d_transpose(conv3, use_bias=False, filters=channel*4, \
        #        kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn4'))
        deconv5 = tf.nn.relu(batch_normalization(tf.layers.conv2d_transpose(feature_up, use_bias=False, filters=channel*2, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn5'))
        deconv6 = tf.nn.relu(batch_normalization(tf.layers.conv2d_transpose(deconv5, use_bias=False, filters=channel, \
                kernel_size=[3, 3], strides=2, padding='SAME'), training=training, scope='bn6'))
        #kernel_size[5, 5] for 8x down sample
        final = tf.nn.sigmoid(tf.layers.conv2d(deconv6, use_bias=False, filters=1, kernel_size=[1, 1], padding='VALID'))

    return feature_out, final

def train(learning_rate = 0.001):
    # tf placeholder
    input_x = tf.placeholder(tf.float32, [None, 28, 28, 1]) 
    LR = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    feature_gap, decoded = AutoEncoder(input_x, training)
    loss = tf.losses.mean_squared_error(labels=input_x, predictions=decoded)
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * weight_decay
    train_op = tf.train.AdamOptimizer(LR).minimize(loss + l2_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    mnist = input_data.read_data_sets('/disk2/mnist_challenge/MNIST_data/', one_hot=False)     # use not one-hotted target data

    steps = mnist.train.num_examples // batch_size
    for epoch in range(epoches):
        if epoch in [10,]:
            learning_rate /= 10
        for step in range(steps):
            train_data, _ = mnist.train.next_batch(batch_size)
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            _, mse_loss, weight_loss = sess.run([train_op, loss, l2_loss], \
                feed_dict={input_x: train_data, LR: learning_rate, training: True})
            
            val_loss = 0
            for i in range(mnist.test.num_examples // 1000):
                val_data, _ = mnist.test.next_batch(1000)
                val_data = np.reshape(val_data, (-1, 28, 28, 1))
                tmp = sess.run([loss], {input_x: val_data, training: False})
                val_loss += tmp[0]
            val_loss /= mnist.test.num_examples // 1000

            if step % 100 == 0:
                print('epoch:{} step:{} mse_loss:{:.4f} l2_loss:{:.4f} val_loss:{:.4f}'.format(epoch, step, mse_loss, weight_loss, val_loss))
        saver.save(sess=sess, save_path=model_path + 'AutoEncoder.ckpt')
    
    val_labels = []
    all_feature = []
    for i in range((mnist.test.num_examples//1000)+1):
        val_data, val_label = mnist.test.next_batch(1000)
        val_data = np.reshape(val_data, (-1, 28, 28, 1))
        feature = sess.run([feature_gap], {input_x: val_data, training: False})
        val_labels.append(val_label)
        all_feature.append(np.array(feature).squeeze())
    val_labels = np.concatenate(val_labels, axis=0)
    all_feature = np.concatenate(all_feature, axis=0)
    tsne = TSNE()
    res = tsne.fit_transform(all_feature)
    plt.scatter(res[:,0], res[:, 1], c=val_labels)
    plt.savefig('res.png')


# For mnist dataset, function predict is not used.
def predict():
    # tf placeholder
    input_x = tf.placeholder(tf.float32, [None, 100, 100, 1])    # value in the range of (0, 1)
    training = tf.placeholder(tf.bool)
    feature_gap, logits = AutoEncoder(input_x, training)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, model_path + 'AutoEncoder.ckpt')
    

    data_dir = '/disk2/dataset/UrbanRegionFunctionClassification/baseline/train/'
    files = os.listdir(data_dir)
    images = []
    names = []
    labels = []
    for f in files:
        img_path = data_dir + '/' + f
        img = cv2.imread(img_path)
        images.append(img.reshape(1, image_size, image_size, img_channels))
        names.append(f)
        labels.append([int(f[-5])])
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    images = images[..., 0] * 0.114 + images[..., 1] * 0.587 + images[..., 2] * 0.299  #cv2.imread -> BGR
    images = images[..., np.newaxis] / 255
    
    all_feature = []
    all_prediction = []
    for i in range((images.shape[0]//1000)+1):
        feature, prediction = sess.run([feature_gap, logits], {input_x: images[i*1000 : min(i*1000+1000, images.shape[0])], training: False})
        all_feature.append(feature)
        all_prediction.append(prediction)
    all_feature = np.concatenate(all_feature, axis=0)
    all_prediction = np.concatenate(all_prediction, axis=0)
    print('predict successfully') 
    for i in range(all_feature.shape[0]):
        with open('feature_5.txt','a') as f:
            for j in range(all_feature.shape[1]):
                f.write(str(all_feature[i,j])+' ')
            f.write(str(labels[i])+' ')
            f.write(names[i]+'\n')
    tsne = TSNE()
    res = tsne.fit_transform(all_feature)
    for i in range(res.shape[0]):
        with open('tsne_5.txt','a') as f:
            f.write(str(res[i,0])+' '+str(res[i,1])+' '+str(labels[i])+' '+names[i]+'\n')
    tsne = TSNE()
    res = tsne.fit_transform(all_prediction)
    for i in range(res.shape[0]):
        with open('tsne_logits_5.txt','a') as f:
            f.write(str(res[i,0])+' '+str(res[i,1])+' '+str(labels[i])+' '+names[i]+'\n')
        
if __name__ == '__main__':

    if len(sys.argv) > 1:
        predict()
    else:
        train()
