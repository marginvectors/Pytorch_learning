import torch

def init_adam_states(feature_dim):
    v_w,v_b = torch.zeros((feature_dim,1)),torch.zeros(1)
    s_w,s_b = torch.zeros((feature_dim,1)),torch.zeros(1)
    return ((v_w,s_w),(v_b,s_b))

def adam(params,states,hyperparams):
    beta1,beta2,eps = 0.9,0.999,1e-6
    for p,(v,s) in zip(params,states):
        with torch.no_grad():
            v[:] = beta1 * v + (1-beta1)*p.grad
            s[:] = beta2 * s + (1-beta2)*torch.square(p.grad)
            v_bias_corr = v / (1-beta1 ** hyperparams['t'])
            s_bias_corr = s / (1-beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)+eps)
        
        p.grad.data.zero_()
    hyperparams['t'] += 1


