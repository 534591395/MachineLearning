import numpy as np 
import tensorflow as tf
#import matplotlib.pyplot as plt
#%matplotlib inline

data_x = np.linspace(0,10,30)
data_y = data_x * 3 + 7 + np.random.normal(0,1,30)

#plt.scatter(data_x, data_y)

#print(data_y)


# 1. 定义参数；2.输入训练数据；3.执行推断；4.计算损失；5.训练模型，减少损失；6.评估

# y = w*x + b;
#定义参数
w = tf.Variable(1., name='quanzhong')
b = tf.Variable(0., name='pianzhi')

# 占位符
# 输入训练数据
x = tf.placeholder(tf.float32, shape=None)
y = tf.placeholder(tf.float32, shape=[None])

# 执行推断 
# pred = x * w + b
pred = tf.multiply(x, w) + b

# 计算损失
# 计算推断y值和点y值之间的平方差后，相加
loss = tf.reduce_sum(tf.squared_difference(pred, y))

# 训练模型
# 梯度下降算法
# 学习速率
learn_rate = 0.0001
# 调用 GradientDescentOptimizer 梯度下降算法，目标为最小化loss
# 模型
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# 训练
sess = tf.Session()
# 初始化变量
sess.run(tf.global_variables_initializer())

# 训练1万次
for i in range(10000):
  sess.run(train_step, feed_dict={x:data_x,y:data_y})
  if i%1000 == 0:
    print(sess.run([loss,w,b], feed_dict={x:data_x,y:data_y}))

print(sess.run(12*w+b))
tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
graph = sess.graph
print([node.name for node in graph.as_graph_def().node])
builder = tf.saved_model.builder.SavedModelBuilder('E:\program\Learn\机器学习\实践\线性回归\Model\saved_model')
builder.add_meta_graph_and_variables(sess, ['tag_string'])
builder.save()


#writer = tf.summary.FileWriter('D://log', sess.graph);