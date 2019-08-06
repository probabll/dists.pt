import torch
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl, kl_divergence
import torch.nn.functional as F


EPS = 1e-5


class MixtureD0C01(torch.distributions.Distribution):
    
    def __init__(self, logits0, cont, validate_args=None):
        """
        - with probability p_0 = sigmoid(logits0) this returns 0
        - with probability 1 - p_0 this returns a sample in the open interval (0, 1)
        
        logits0: logits for p_0
        cont: a (properly normalised) distribution over (0, 1)
            e.g. RightTruncatedExponential
        """
        shape = cont.batch_shape
        super(MixtureD0C01, self).__init__(batch_shape=shape, validate_args=validate_args)
        self.logits = logits0
        self.cont = cont
        self.log_p0 = F.logsigmoid(self.logits)
        self.p0 = torch.sigmoid(self.logits)        
        self.pc = 1. - self.p0
        self.log_pc = - F.softplus(logits0)  # = torch.log(self.pc)
        self.uniform = Uniform(torch.zeros(shape).to(logits0.device), 
                               torch.ones(shape).to(logits0.device))
                
    def rsample(self, sample_shape=torch.Size()): 
        # sample from (0, 1) uniformly
        u = self.uniform.rsample(sample_shape)  
        # affine transform to project from (p_0, 1) to (0, 1)
        # note that only where p_0 < u < 1 this is correct
        to_cont = (u - self.p0) / self.pc    
        # c ~ ContinuousDist()
        # note where p_0 < u < 1, c is valid and is in (0,1)
        c = self.cont.icdf(to_cont)
        # inverse cdf of mixture model
        # 0 if u < p_0
        # c otherwise
        x = torch.where(u <= self.p0, torch.zeros_like(u), c)
        return x
    
    def log_prob(self, value):    
        log_prob_cont = self.cont.log_prob(value)
        log_prob = torch.where(value == 0., self.log_p0, self.log_pc + log_prob_cont)
        return log_prob
    
    def cdf(self, value):
        cdf_cont = self.cont.cdf(value)
        cdf = torch.where(value == 0., self.p0, self.p0 + self.pc * cdf_cont)
        return cdf
    
    def entropy(self):
        h = self.p0 * ( - self.log_p0) + self.pc * (- self.log_pc) + self.pc * self.cont.entropy()
        return h


class MixtureD01C01(torch.distributions.Distribution):

    def __init__(self, logits, cont, validate_args=None):
        """

        logits: [..., 3]
        cont: a (properly normalised) distribution over (0, 1)
            e.g. RightTruncatedExponential
        """
        shape = cont.batch_shape
        super(MixtureD01C01, self).__init__(batch_shape=shape, validate_args=validate_args)
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.logprobs = F.log_softmax(logits, dim=-1)
        self.cont = cont
        self.p0, self.p1, self.pc = [t.squeeze(-1) for t in torch.split(self.probs, 1, dim=-1)]
        self.log_p0, self.log_p1, self.log_pc = [t.squeeze(-1) for t in torch.split(self.logprobs, 1, dim=-1)]
        self.uniform = Uniform(torch.zeros(shape).to(logits.device),
                               torch.ones(shape).to(logits.device))

    def rsample(self, sample_shape=torch.Size()):
        # sample from (0, 1) uniformly
        u = self.uniform.rsample(sample_shape)
        # affine transform to project from (p_0, 1) to (0, 1)
        # note that only where p_0 < u < 1 this is correct
        # print(u.size(), self.p0.size(), self.pc.size())
        to_cont = (u - self.p0) / self.pc
        # c ~ ContinuousDist()
        # note where p_0 < u < 1, c is valid and is in (0,1)
        x = self.cont.icdf(to_cont)
        # inverse cdf of mixture model
        # 0 if u < p_0
        # c otherwise
        x = torch.where(u <= self.p0, torch.zeros_like(u), x)
        x = torch.where(u >= self.p0 + self.pc, torch.ones_like(u), x)
        return x

    def log_prob(self, value):
        log_prob = self.log_pc + self.cont.log_prob(value)
        log_prob = torch.where(value == 0., self.log_p0, log_prob)
        log_prob = torch.where(value == 1., self.log_p1, log_prob)
        return log_prob

    def cdf(self, value):
        cdf = self.p0 + self.pc * self.cont.cdf(value)
        cdf = torch.where(value == 0., self.p0, cdf)
        cdf = torch.where(value == 1., self.p1, cdf)
        return cdf

    def entropy(self):
        h = - self.p0 * self.log_p0 - self.p1 * self.log_p1 - self.pc * self.log_pc + self.pc * self.cont.entropy()
        return h

    
def kl_mixture_mixture(p, q):   
    # see derivation on overleaf
    kl = p.p0 * (p.log_p0 - q.log_p0)
    kl = kl + p.pc * (p.log_pc - q.log_pc)
    kl = kl + p.pc * kl_divergence(p.cont, q.cont)
    return kl


@register_kl(MixtureD0C01, MixtureD0C01)
def _kl(p, q):
    return kl_mixture_mixture(p, q)