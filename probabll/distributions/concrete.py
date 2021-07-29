import torch
from torch.distributions.uniform import Uniform
from torch.distributions.kl import register_kl
import torch.distributions as td

EPS = 1e-5

class BinaryConcrete(torch.distributions.relaxed_bernoulli.RelaxedBernoulli):
    
    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super(BinaryConcrete, self).__init__(temperature, probs=probs, logits=logits, validate_args=validate_args)
        
    def cdf(self, value):
        return torch.sigmoid((torch.log(value + EPS) - torch.log(1. - value + EPS)) * self.temperature - self.logits)
    
    def icdf(self, value):
        return torch.sigmoid((torch.log(value + EPS) - torch.log(1. - value + EPS) + self.logits) / self.temperature)
        
    def rsample_truncated(self, k0, k1, sample_shape=torch.Size()):        
        shape = self._extended_shape(sample_shape)
        probs = torch.distributions.utils.clamp_probs(self.probs.expand(shape))
        uniforms = Uniform(self.cdf(torch.full_like(self.logits, k0)), 
                           self.cdf(torch.full_like(self.logits, k1))).rsample(sample_shape)
        x = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / self.temperature
        return torch.sigmoid(x)


def kl_concrete_concrete(p, q, n_samples=1):
    """
    KL is estimated for the logits of the concrete distribution to avoid underflow.
    """
    x_logit = p.base_dist.rsample(torch.Size([n_samples]))
    return (p.base_dist.log_prob(x_logit) - q.base_dist.log_prob(x_logit)).mean(0)


@register_kl(BinaryConcrete, BinaryConcrete)
def _kl_concrete_concrete(p, q):
    return kl_concrete_concrete(p, q, n_samples=1)


class RelaxedOneHotCategoricalStraightThrough(td.RelaxedOneHotCategorical):

    arg_constraints = {
        'probs': td.constraints.simplex,
        'logits': td.constraints.real_vector
    }
    support = td.constraints.simplex
    has_rsample = True

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        super().__init__(temperature, probs=probs, logits=logits, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedOneHotCategoricalStraightThrough, _instance)
        return super(RelaxedOneHotCategoricalStraightThrough, self).expand(batch_shape, _instance=new)

    def _project(self, relaxed_value):
        shape = relaxed_value.size()
        _, ind = relaxed_value.max(dim=-1)
        hard = torch.zeros_like(relaxed_value).view(-1, shape[-1])
        hard.scatter_(1, ind.view(-1, 1), 1)
        hard = hard.view(*shape)
        return hard

    def sample(self, sample_shape=torch.Size()):
        return self._project(super().sample(sample_shape))

    def rsample(self, sample_shape=torch.Size()):
        sample = super().rsample(sample_shape)
        hard = self._project(sample)
        return (hard - sample).detach() + sample

    def entropy(self):
        return td.Categorical(logits=self.logits).entropy()

    def log_prob(self, value):
        return td.OneHotCategorical(logits=self.logits).log_prob(value)

    def enumerate_support(self, expand=True):
        return td.OneHotCategorical(logits=self.logits).enumerate_support(expand=expand)


@register_kl(RelaxedOneHotCategoricalStraightThrough, RelaxedOneHotCategoricalStraightThrough)
def _kl_concretest_concretest(p, q):
    return td.kl_divergence(td.Categorical(logits=p.logits), td.Categorical(logits=q.logits))

