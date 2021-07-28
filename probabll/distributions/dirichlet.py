import torch
import torch.distributions as td
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from itertools import product
import numpy as np


class MaskedDirichlet(td.Distribution):
    """
    This class allows us to batch Dirichlet distributions of varying dimensionality.
    """

    arg_constraints = {'concentration': td.constraints.independent(td.constraints.greater_than_eq(0.), 1)}
    support = td.constraints.simplex
    has_rsample = True

    def __init__(self, mask, concentration, validate_args=None):
        assert mask.shape == concentration.shape, f"Got mask {mask.shape} and concentration {concentration.shape}"

        if validate_args:
            if not (mask.sum(-1) > 0).all():
                raise ValueError("A face must contain at least one vertix (check that every row in 'mask' contains at least one coordinate set to True)")    
            if not (torch.where(mask, concentration, torch.ones_like(concentration)) > 0).all():
                raise ValueError("The concentration parameter must be strictly positive for positions where mask is True")
        
        self._mask = mask
        self._concentration = concentration

        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super(MaskedDirichlet, self).__init__(
            batch_shape, 
            event_shape, 
            validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MaskedDirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new._concentration = self._concentration.expand(batch_shape + self.event_shape)
        new._mask = self._mask.expand(batch_shape + self.event_shape)
        super(MaskedDirichlet, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # X_k | f ~ Gamma(alpha_{f,k}, 1)
        X = td.Gamma(self._concentration, torch.ones_like(self._concentration), validate_args=False)  
        # [batch_size, K]
        x = X.rsample(sample_shape)
        # now we mask the Gamma samples from invalid coordinates of lower-dimensional faces
        x = torch.where(self._mask, x, torch.zeros_like(x))  
        # finally, we renormalise the gamma samples
        z = x / x.sum(-1, keepdim=True)
        return z

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)        

        # logarithm of Dirichlet normaliser (log Beta(alpha))       
        zeros = torch.zeros_like(self._concentration)
        log_beta_num = torch.where(self._mask, torch.lgamma(self._concentration), zeros).sum(-1)
        log_beta_den = torch.lgamma(torch.where(self._mask, self._concentration, zeros).sum(-1))
        log_beta = log_beta_num - log_beta_den

        # logarithm of unnormalised density
        log_prob = torch.where(self._mask, (self._concentration - 1) * torch.log(value), torch.zeros_like(value)).sum(-1)

        # normalisation
        log_prob = log_prob - log_beta
        return log_prob

    @property
    def concentration(self):
        """Masked concentration parameters"""
        return torch.where(self._mask, self._concentration, torch.zeros_like(self._concentration))

    @property
    def mask(self):
        return self._mask

    @property
    def dim(self):
        """Dimensionality of each face as a tensor of dtype=int64 and shape=batch_shape"""
        return self._mask.sum(-1)

    @property
    def mean(self):        
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def variance(self):
        concentration = self.concentration
        con0 = concentration.sum(-1, True)
        return concentration * (con0 - concentration) / (con0.pow(2) * (con0 + 1))

    def entropy(self):
        # dimensionality of each Dirichlet
        K = self._mask.float().sum(-1)
        # masked concentrations
        concentration = self.concentration
        a0 = concentration.sum(-1)
        zeros = torch.zeros_like(self._concentration)
        H = torch.where(self._mask, torch.lgamma(self._concentration), zeros).sum(-1) 
        H = H - torch.lgamma(a0) 
        H = H - (K - a0) * torch.digamma(a0)
        H = H - (torch.where(self._mask, (self._concentration - 1.0) * torch.digamma(self._concentration), zeros).sum(-1))
        return H

    def cross_entropy(self, other):
        return self.entropy() + td.kl_divergence(self, other)


@td.register_kl(MaskedDirichlet, MaskedDirichlet)
def _kl_maskeddirichlet_maskeddirichlet(p, q):
    if p.mask.shape != q.mask.shape: 
        raise ValueError("The shapes of p and q differ")
    if not (p.mask == q.mask).all():
        raise ValueError("The faces in p and q differ")
    
    p_concentration = p.concentration
    q_concentration = q.concentration

    sum_p_concentration = p_concentration.sum(-1)
    sum_q_concentration = q_concentration.sum(-1)
    t1 = sum_p_concentration.lgamma() - sum_q_concentration.lgamma()
    
    zeros = torch.zeros_like(p_concentration)
    t2 = torch.where(p.mask, p_concentration.lgamma() - q_concentration.lgamma(), zeros).sum(-1)
    t3 = p_concentration - q_concentration
    t4 = torch.where(p.mask, p_concentration.digamma() - sum_p_concentration.digamma().unsqueeze(-1), zeros)
    return t1 - t2 + (t3 * t4).sum(-1)


### Test stuff

def make_faces(K, dtype=bool, device=torch.device('cpu')):
    """
    Return a bit-vector representation of all non-empty faces of 
    K-1 dimensional simplex. 

    The return type is an np.array with dtype=bool and shape=[2^K-1, K]
    """
    faces = [x for x in product([0, 1], repeat=K) if sum(x)]
    return torch.tensor(np.array(faces, dtype=dtype), device=device)


def get_parameters(faces, dir_alpha=1, gamma_alpha=1, gamma_beta=1, device=torch.device('cpu')):
    """
    Return tf trainable parameters for a stratified distribution defined over the given faces
        * mixing coefficients (omega) with shape [num_faces]
        * Gamma concentrations (alpha) with shape [num_faces, K]

    :param faces: bit-vector encoding of faces [num_faces, K]
    :param dir_alpha: Dir concentration (float) used to sample the initial value of omega
        set it to None to get omega_k = 1/K
        set it to 'face_dim' to set each concentration parameter to the dimensionality of the face
        this will bias the initial omega to be higher for higher-dimensional faces
    :param gamma_alpha: Gamma shape parameter used to sample the initial values of alpha
    :param gamma_beta: Gamma rate parameter used to sample the initial values of alpha
    """
    if dir_alpha is None:  # uniform mixing coefficients
        omega = torch.tensor(np.ones(faces.shape[0])/faces.shape[0], requires_grad=True, device=device)
    elif isinstance(dir_alpha, str) and dir_alpha == 'face_dim':
        omega = torch.tensor(np.random.dirichlet(faces.float().sum(-1)), requires_grad=True, device=device)
    elif isinstance(dir_alpha, float) or isinstance(dir_alpha, int):
        omega = torch.tensor(np.random.dirichlet(np.ones(faces.shape[0]) * dir_alpha), requires_grad=True, device=device)
    else:
        raise ValueError("Use None, 'face_dim', or a strictly positive real for dir_alpha")
    alphas = torch.tensor(np.random.gamma(gamma_alpha, 1/gamma_beta, size=faces.shape), requires_grad=True, device=device)
    return omega, alphas

def test_masked_dirichlet(K=3):
    mask = make_faces(K)
    w, con = get_parameters(mask, dir_alpha=None, gamma_alpha=1, gamma_beta=1)
    p = MaskedDirichlet(mask, con)
    q = MaskedDirichlet(mask, torch.ones_like(con))
    assert (torch.where(torch.logical_not(mask), p.concentration, torch.zeros_like(con)) == 0).all(), "Masked concentration parameters should be 0.0"
    assert (torch.where(mask, p.concentration, torch.ones_like(con)) > 0).all(), "Unmasked concentration parameters should be strictly positive"
    for i, face in enumerate(mask):
        idx = tuple(k for k, b in enumerate(face) if b)
        alphas = con[i,idx]
        p_low = td.Dirichlet(alphas)
        q_low = td.Dirichlet(torch.ones_like(alphas))
        assert torch.isclose(p.mean[i,idx], p_low.mean).all(), f"The {i}th face's mean does not match that of td.Dirichlet"
        assert torch.isclose(p.variance[i,idx], p_low.variance).all(), f"The {i}th face's variance does not match that of td.Dirichlet"
        assert torch.isclose(p.entropy()[i], p_low.entropy()).all(), f"The {i}th face's entropy does not match that of td.Dirichlet"
        assert (p.dim[i] == len(idx)), f"The dimensionality of the {i}th face is incorrect: got {p.dim[i]}, expected {len(idx)}"
        x = p.rsample()
        assert torch.isclose(p.log_prob(x)[i], p_low.log_prob(x[i,idx])).all(), "The log_prob of a sample does not match that assigned by td.Dirichlet"
        assert torch.isclose(td.kl_divergence(p, q)[i], td.kl_divergence(p_low, q_low)).all(), "The KL divergence does not match that of td.Dirichlet"
        assert torch.isclose(td.kl_divergence(q, p)[i], td.kl_divergence(q_low, p_low)).all(), "The KL divergence does not match that of td.Dirichlet"


test_masked_dirichlet(3)
test_masked_dirichlet(5)
test_masked_dirichlet(7)


