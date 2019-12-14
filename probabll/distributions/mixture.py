import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Categorical, Independent
from torch.distributions import register_kl


class MixtureSameFamily(Distribution):
    """
    A distribution made of a discrete mixture of K distributions of the same family.

    Consider the example of mixing K diagonal Gaussians:
    
        X|\phi, \mu, \sigma ~ MixtureSameFamily(\phi, Independent(Normal(\mu, \sigma^2), 1))
    
    where
        * \phi \in R^K
        * \mu \in R^D
        * \sigma \in R+^D
        
    is such that 
        I|\phi ~ Categorical(softmax(\phi))
        X|i, \mu_i, \sigma_i ~ Normal(\mu_i, \sigma_i^2)
        
    Thus, where w = softmax(\phi), 
        p(x) = \sum_{i=1}^K w_i N(x|\mu_i, \sigma_i^2)

    We can sample efficiently (though not yet with a reparameterisation).
    And we can assess log p(x) in closed-form.
    """
    
    def __init__(self, logits, components: Distribution):
        """
        If your distribution is say a product of D independent Normal variables, make sure to
            wrap it around Independent.
        
        num_components: K
        logits: [B, K] 
        components: [B, K, D] where batch_shape is [K] and event_shape is [D]
            Note that if you have Normal(loc, scale) where the parameters are [K,D]
            you need to wrap it around Independent(Normal(loc, scale), 1) to make the event_shape be [D]
            otherwise it will be []
        """
        
        if len(logits.shape) != len(components.batch_shape):
            raise ValueError("The shape of logits must match the batch shape of your components")
        if logits.shape[-1] != components.batch_shape[-1]:
            raise ValueError("You need as many logits as you have components") 
        # Exclude the component dimension
        batch_shape = logits.shape[:-1]
        num_components = logits.shape[-1]  # K
        super().__init__(batch_shape, components.event_shape)
                                     
        self.num_components = num_components
        self.log_weights = F.log_softmax(logits, dim=-1)
        self.categorical = Categorical(logits=logits)
        self.components = components

    def log_prob(self, x):
        """
        x: [sample_shape, batch_shape, event_shape]
        returns: [sample_shape, batch_shape]
        """
        # Let's introduce a dimension for the components
        # [sample_shape, batch_shape, 1, event_shape]
        x = x.unsqueeze(-len(self.event_shape) - 1)
        # [sample_shape, batch_shape, num_components]
        log_joint_prob = self.components.log_prob(x) + self.log_weights
        # now we marginalise the components
        return torch.logsumexp(log_joint_prob, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError("Not implemented yet")

    def sample(self, sample_shape=torch.Size()): 
        """Return a sample with shape [sample_shape, batch_shape, event_shape]"""
        # [sample_shape, batch_shape, num_components, event_shape]
        x = self.components.rsample(sample_shape)
        # [sample_shape, batch_shape]
        indicators = self.categorical.sample(sample_shape)
        # [sample_shape, batch_shape, num_components]
        indicators = F.one_hot(indicators, self.num_components)
        if len(self.components.event_shape):
            # [sample_shape, batch_shape, num_components, 1]
            indicators = indicators.unsqueeze(-1)
        # reduce the component dimension
        return (x * indicators.type(x.dtype)).sum(len(sample_shape) + len(self.batch_shape))  


class MixtureOfGaussians(MixtureSameFamily):

    def __init__(self, logits, locations, scales):
        """
        logits: [B, K]
        locations: [B, K, D]
        scales: [B, K, D]
        """
        super().__init__(logits, Independent(Normal(loc=locations, scale=scales), 1))
    

def kl_gaussian_mog(p, q, num_samples=1):
    """
    Estimate KL(p||q) = E_p[ \log q(z) - log p(z)] = E_p[\log p(z)] - E_p[log q(z)] = - H(p) - E_p[log q(z)]
        where we either use the closed-form entropy of p if available, or MC estimate it, 
        and always MC-estimate the second term.
    """
    if num_samples == 1:
        # [1, ...]
        z = p.rsample().unsqueeze(0)
    else:
        # [num_samples, ...]
        z = p.rsample(torch.Size([num_samples]))
    try:
        H_p = p.entropy()
    except NotImplementedError:
        H_p = - p.log_prob(z).mean(0)

    return - H_p - q.log_prob(z).mean(0)

@register_kl(Independent, MixtureSameFamily)
def _kl_gaussian_mog(p, q):
    return kl_gaussian_mog(p, q, 1)

