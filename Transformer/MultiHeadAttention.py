def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim,1))
    v_b = torch.zeros(1)
    return (v_w,v_b)

def sgd_momentum(params,states,hyperparams):
    for p,v in zip(params,states):
        with torch.no_grad():
            v[:]=hyperparams['momentum']*v +(1-hyperparams['momentum'])*p.grad
            p[:]-= hyperparams['lr'] * v
        p.grad.data.zero_()

# MindSpore
class Momentum(nn.Optimizer):
    """定义优化器"""
    def __init__(self, params, learning_rate, momentum=0.9):
        super(Momentum, self).__init__(learning_rate, params)
        self.momentum = Parameter(Tensor(momentum, ms.float32), name="momentum")
        self.moments = self.parameters.clone(prefix="moments", init="zeros")

    def construct(self, gradients):
        """construct输入为梯度，在训练中自动传入梯度gradients"""
        lr = self.get_lr()
        params = self.parameters # 待更新的权重参数

        for i in range(len(params)):
            # 更新moments值
            ops.assign(self.moments[i], self.moments[i] * self.momentum + (1-self.momentum)*gradients[i])
            update = params[i] - self.moments[i] * lr  #带有动量的SGD算法
            ops.assign(params[i], update)
        return params

net = Net()
# 设置优化器待优化的参数和学习率为0.01
opt = Momentum(net.trainable_params(), 0.01)
