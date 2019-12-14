import torch
from torch.distributions import constraints, Beta, Uniform 
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all
from torch.distributions.gumbel import euler_constant
from torch.distributions.kl import register_kl


EPS = 1e-5

def _lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

def _harmonic_number(x):
    return torch.digamma(x + 1) + euler_constant
    

class Kumaraswamy(TransformedDistribution):
    r"""
    Samples from a two-parameter Kumaraswamy distribution with a, b parameters. Or equivalently,
        U ~ U(0,1)
        X = (1 - (1 - U)^(1 / b))^(1 / a)

    Example:

        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kuma distribution with a=1, n=1
        tensor([ 0.4784])

    Args:
        a (float or Tensor): TODO.
        b (float or Tensor): TODO.
    """
    arg_constraints = {'a': constraints.positive, 'b': constraints.positive}
    support = constraints.unit_interval

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        self.a_reciprocal = self.a.reciprocal()
        self.b_reciprocal = self.b.reciprocal()
        base_dist = Uniform(torch.full_like(self.a, EPS), torch.full_like(self.a, 1. - EPS))
        transforms = [AffineTransform(loc=1, scale=-1),
                      PowerTransform(self.b_reciprocal),
                      AffineTransform(loc=1, scale=-1),
                      PowerTransform(self.a_reciprocal)]

        super(Kumaraswamy, self).__init__(base_dist,
                                          transforms,
                                          validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Kuma, _instance)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        new.a_reciprocal = new.a.reciprocal()
        new.b_reciprocal = new.b.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        transforms = [AffineTransform(loc=1, scale=-1),
                      PowerTransform(self.b_reciprocal),
                      AffineTransform(loc=1, scale=-1),
                      PowerTransform(self.a_reciprocal)]
        super(Kumaraswamy, new).__init__(base_dist,
                                         transforms,
                                         validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        return torch.log(self.a + EPS) + torch.log(self.b + EPS) + (self.a - 1) * torch.log(value + EPS) \
            + (self.b - 1) * torch.log(1 - value ** self.a + EPS)

    #def log_cdf(self, value):        
    #    return torch.log(1. - (1. - value ** self.a + EPS) ** self.b + EPS) 
    
    def cdf(self, value):
        """
        cdf: torch.exp(torch.log1p(-(1 - value ** self.a) ** self.b))
        icdf: torch.exp(torch.log1p(-torch.exp(torch.log1p(-value) / self.b)) / self.a)
        """
        return torch.exp(torch.log1p(-(1 - value ** self.a) ** self.b))
        #return torch.exp(self.log_cdf(torch.clamp(value, min=EPS, max=1-EPS)))


    @property
    def mean(self):
        return self._moment(1)

    @property
    def variance(self):
        return self._moment(2) - self._moment(1) ** 2

    def entropy(self):
        return (1 - self.b_reciprocal) + (1 - self.a_reciprocal) * _harmonic_number(self.b) \
                - torch.log(self.a) - torch.log(self.b)
    
    def _moment(self, n):
        return self.b * torch.exp(_lbeta(1 + n * self.a_reciprocal, self.b))    

    def rsample_truncated(self, k0, k1, sample_shape=torch.Size()):
        """
        Sample over a truncated support
        """
        # U ~ Uniform(cdf(k0), cdf(k1)) 
        # K = F^-1(U)
        #  simulates K over the truncated support (k0,k1)
        
        l_b = torch.clamp(self.cdf(torch.full_like(self.a, k0)), min=EPS)
        h_b = torch.clamp(self.cdf(torch.full_like(self.b, k1)), max=(1 - EPS*10))
        
        x = Uniform(l_b, h_b).rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

def kl_kumaraswamy_kumaraswamy(p, q, n_samples=1, exact_entropy=True):
    """    
    KL(p||q) = -H(p) + H(q|p)
     where the entropy can be computed in closed form or estimated
     the cross entropy is always estimated
    """
    x = p.rsample(sample_shape=torch.Size([n_samples]))
    if exact_entropy:
        p_entropy = p.entropy()
    else:
        p_entropy = - p.log_prob(x).mean(0)
    cross_entropy = - q.log_prob(x).mean(0)
    return - p_entropy + cross_entropy


@register_kl(Kumaraswamy, Kumaraswamy)
def _kl_kumaraswamy_kumaraswamy(p, q):
    return kl_kumaraswamy_kumaraswamy(p, q, n_samples=1, exact_entropy=True)


@register_kl(Kumaraswamy, Uniform)
def _kl(p, q):
    # E_p[log U] is a constant
    #  the constant corresponds to 1 if U is U(0,1)
    #  to be sure we compute it over the (0, 1) interval in terms of the Uniform's cdf
    constant = - torch.log(q.cdf(1) - q.cdf(0))
    return - p.entropy() - constant


@register_kl(Kumaraswamy, Beta)
def _kl_kumaraswamy_beta(p, q, m=10):
    """
    Return KL(Kumaraswamy(a,b) || Beta(alpha, beta)). We employ the approximation by https://arxiv.org/pdf/1605.06197.pdf
    p: a (batch of) Kumaraswamy distribution(s)
    q: a (batch of) Beta distribution(s)
    m: number of terms in the taylor expansion
    """
    # This is a Kumaraswamy
    pa = p.a
    pb = p.b
    # The Beta in torch.distributions has different parameter names
    qa = q.concentration1
    qb = q.concentration0
    # KL approximation
    term1 = (pa - qa) / pa * (- euler_constant - torch.digamma(pb) - torch.reciprocal(pb))
    term1 += torch.log(pa) + torch.log(pb) + _lbeta(qa, qb)
    term1 -= (pb - 1) / pb
    # Truncated Taylor expansion around 1
    log_taylor = torch.logsumexp(torch.stack([_lbeta(m / pa, pb) - torch.log(m + pa * pb) for m in range(1, 10 + 1)], dim=-1), dim=-1)
    term2 = (qb - 1) * pb * torch.exp(log_taylor)
    # We truncate the expansion from below just in case
    return torch.clamp(term1 + term2, min=0.)

