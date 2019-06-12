import torch

def mc_entropy(p, n_samples=1):
    x = p.rsample(sample_shape=torch.Size([n_samples]))
    return - p.log_prob(x).mean(0)


def mc_kl(p, q, n_samples=1, exact_entropy=False):
    x = p.rsample(sample_shape=torch.Size([n_samples]))
    if exact_entropy:
        p_entropy = p.entropy()
    else:
        p_entropy = - p.log_prob(x).mean(0)
    cross_entropy = - q.log_prob(x).mean(0)
    return - p_entropy + cross_entropy