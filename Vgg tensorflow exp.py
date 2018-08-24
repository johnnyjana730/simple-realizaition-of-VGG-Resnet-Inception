import tensorflow as tf
import numpy as np
# import TFRecord
 
# parameter
learning_rate = 0.001
display_step = 5
epochs = 10
keep_prob = 0.5


# fc
def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                shape = [n_in, n_out],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope) 
        return activation

#case conv1_1``
ssd_shape = np.zeros((548,548),dtype=int)
print
images = tf.Variable(tf.random_normal([2, ssd_shape.shape[0],ssd_shape.shape[1], 3]))

filter = tf.Variable(tf.random_normal([3,3,3,64]))
net = tf.nn.conv2d(images, filter, strides=[1, 1, 1, 1], padding='VALID')
filter2 = tf.Variable(tf.random_normal([3,3,64,64]))
net = tf.nn.conv2d(net, filter2, strides=[1, 1, 1, 1], padding='VALID')
net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='VALID')
#case conv2`
filter3 = tf.Variable(tf.random_normal([3,3,64,128]))
net = tf.nn.conv2d(net, filter3, strides=[1, 1, 1, 1], padding='VALID')
filter4 = tf.Variable(tf.random_normal([3,3,128,128]))
net = tf.nn.conv2d(net, filter4, strides=[1, 1, 1, 1], padding='VALID')
net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='VALID')
#case conv3
filter5 = tf.Variable(tf.random_normal([3,3,128,256]))
net = tf.nn.conv2d(net, filter5, strides=[1, 1, 1, 1], padding='VALID')
filter6 = tf.Variable(tf.random_normal([3,3,256,256]))
net = tf.nn.conv2d(net, filter6, strides=[1, 1, 1, 1], padding='VALID')
filter7 = tf.Variable(tf.random_normal([3,3,256,256]))
net = tf.nn.conv2d(net, filter7, strides=[1, 1, 1, 1], padding='VALID')
net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='VALID')
#case conv4
filter8 = tf.Variable(tf.random_normal([3,3,256,512]))
net = tf.nn.conv2d(net, filter8, strides=[1, 1, 1, 1], padding='VALID')
filter9 = tf.Variable(tf.random_normal([3,3,512,512]))
net = tf.nn.conv2d(net, filter9, strides=[1, 1, 1, 1], padding='VALID')
filter10 = tf.Variable(tf.random_normal([3,3,512,512]))
net = tf.nn.conv2d(net, filter10, strides=[1, 1, 1, 1], padding='VALID')
net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='VALID')
#case conv5
filter11 = tf.Variable(tf.random_normal([3,3,512,512]))
net = tf.nn.conv2d(net, filter11, strides=[1, 1, 1, 1], padding='VALID')
filter12 = tf.Variable(tf.random_normal([3,3,512,512]))
net = tf.nn.conv2d(net, filter12, strides=[1, 1, 1, 1], padding='VALID')
filter13 = tf.Variable(tf.random_normal([3,3,512,512]))
net = tf.nn.conv2d(net, filter13, strides=[1, 1, 1, 1], padding='VALID')
net = tf.nn.max_pool(net,[1,3,3,1],[1,1,1,1],padding='VALID')
# case conv6
filter14 = tf.Variable(tf.random_normal([3,3,512,1024]))
net = tf.nn.conv2d(net, filter14, strides=[1, 1, 1, 1], padding='VALID')
print(net)
# 19
# case conv7
filter15 = tf.Variable(tf.random_normal([1,1,1024,1024]))
net = tf.nn.conv2d(net, filter15, strides=[1, 1, 1, 1], padding='VALID')
# case conv8
filter16 = tf.Variable(tf.random_normal([1,1,1024,256]))
net = tf.nn.conv2d(net, filter16, strides=[1, 1, 1, 1], padding='VALID')
filter17 = tf.Variable(tf.random_normal([3,3,256,512]))
net = tf.nn.conv2d(net, filter17, strides=[1, 2, 2, 1], padding='SAME')
# conv9
filter = tf.Variable(tf.random_normal([1,1,512,128]))
net = tf.nn.conv2d(net, filter, strides=[1, 1, 1, 1], padding='VALID')
filter = tf.Variable(tf.random_normal([3,3,128,256]))
net = tf.nn.conv2d(net, filter, strides=[1, 2, 2, 1], padding='SAME')
# conv10
filter = tf.Variable(tf.random_normal([1,1,256,128]))
net = tf.nn.conv2d(net, filter, strides=[1, 1, 1, 1], padding='VALID')
filter = tf.Variable(tf.random_normal([3,3,128,256]))
net = tf.nn.conv2d(net, filter, strides=[1, 1, 1, 1], padding='SAME')
# conv11
filter = tf.Variable(tf.random_normal([1,1,256,128]))
net = tf.nn.conv2d(net, filter, strides=[1, 1, 1, 1], padding='VALID')
filter = tf.Variable(tf.random_normal([3,3,128,256]))
net = tf.nn.conv2d(net, filter, strides=[1, 1, 1, 1], padding='SAME')

# flatten
shp = net.get_shape()
flattened_shape = shp[1].value * shp[2].value * shp[3].value
resh1 = tf.reshape(net, [-1, flattened_shape], name="resh1")
# fully connected
fc6 = fc_op(resh1, name="fc6", n_out=4096)
fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)
fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
logits = fc_op(fc7_drop, name="fc8", n_out=2)

labels = []
for i in range(2):
    if i < 50:labels.append([1,0])
    else:labels.append([1,0])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)        
    step = 0
    while step < epochs:
        step += 1
        print(step)
        _, loss, acc = sess.run([optimizer,cost,accuracy])
        if step % display_step ==0:  
            print(loss,acc)
    print("training finish!")
    # _, testLoss, testAcc = sess.run([test_optimizer,test_cost,test_acc])
    # print "Test acc = "+ str(testAcc)
