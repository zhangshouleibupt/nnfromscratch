### 随机梯度下降

随机梯度下降，mini-batch 标准梯度下降。

curve fitting说明：

利用多项式和前馈神经网来拟合正弦曲线

练习pytorch中的autograd，可以发现，两者的拟合效果都不是特别好。

其中前馈神经网的拟合如下：

![Figure_1](/media/ciphor/0E5E1A5C5E1A3D41/Figure_1.png)

多项式拟合如下（项数为100）：

![Figure_2](/media/ciphor/0E5E1A5C5E1A3D41/Figure_2.png)

需要注意的是多项式的拟合是有封闭解的，不过我们仍然采用梯度下降的方法来拟合。

而且本例中，我们采用的批量梯度下降的方法。



