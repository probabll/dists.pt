import torch
import torch.nn.functional as F
from torch.distributions.kl import register_kl
from torch.distributions.bernoulli import Bernoulli


@register_kl(Bernoulli, Bernoulli)
def kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (F.softplus(-q.logits) - F.softplus(-p.logits))
    t1[q.probs == 0] = float('inf')
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * (F.softplus(q.logits) - F.softplus(p.logits))
    t2[q.probs == 1] = float('inf')
    t2[p.probs == 1] = 0
    return t1 + t2


def bernoulli_probs_from_logit(logit):
    """
    Let p be the Bernoulli parameter and q = 1 - p.
    This function is a stable computation of p and q from logit = log(p/q).

    :param logit: log (p/q)
    :return: p, q
    """
    return torch.sigmoid(logit), torch.sigmoid(-logit)


def bernoulli_log_probs_from_logit(logit):
    """
    Let p be the Bernoulli parameter and q = 1 - p.
    This function is a stable computation of p and q from logit = log(p/q).

    :param logit: log (p/q)
    :return: log_p, log_q
    """
    return - F.softplus(-logit), - F.softplus(logit)

