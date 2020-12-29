------
查看tensorflow版本

	import tensorflow as tf 
	print(tf.__version__)

### 计算图
首先看一段代码，使用tensorflow输出`hello world`
```python
import tensorflow as tf 
hello = tf.constant("Hello World!")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
sess = tf.Session()
print(sess.run(hello))
# Hello World!
sess.close()
```
tensorflow是依靠图结构存储信息，第一步中是创建一个计算图，第二步是打开一个会话执行计算图

计算a+b 
```python 
import tensorflow as tf 

x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = 'y')
f = x + y 
print(f) # 输出计算图 Tensor("add:0", shape=(), dtype=int32)

# 通过会话执行 
# 写法1 
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
print(sess.run(f))  # 7 
sess.close()

# 写法2
with tf.Session() as sess:
    x.initializer.run()
    sess.run(y.initializer)
    f.eval()

```
一般写法2用的会相对较多,还有一点可以改进的是把为每个变量调用初始化器之外，还可以用global_variables_initializer()函数改进

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
	    init.run()
	    print(ans.eval())

----------------------

### 管理图
你创建的节点会被自动添加到默认图上

	import tensorflow as tf

	x = tf.Variable(3, name='x')
	print(x.graph is tf.get_default_graph())

当你想要管理多个互不依赖的图是，可以创建一个新的图，然后用with临时把这个图设为默认图
```python 
import tensorflow as tf

x1 = tf.Variable(3, name='x1')
print(x1.graph is tf.get_default_graph()) # True

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2,name = 'x2')

print(x2.graph is tf.get_default_graph()) # False
print(x2.graph is graph) # True


x3 = tf.Variable(3, name='x1')
print(x3.graph is tf.get_default_graph()) # True

```


----------------------------
### 节点生命周期



```python 
import tensorflow as tf 

w = tf.constant(3)
x = w +2 
y = x +2 
z = y + 2 
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

```
当计算z的值时，虽然和y有关，且在上一步中已经计算了y的值，但tensorflow并不会调用上一步求的值，也就是图在每次执行时，所有的节点值会被重置，但变量的值不会

```python 
import tensorflow as tf 

w = tf.constant(3)
x = w +2 
y = x +2 
z = y + 2 
with tf.Session() as sess:
    y_val,z_val = sess.run([y,z])
    print(y_val)
    print(z_val)
```
