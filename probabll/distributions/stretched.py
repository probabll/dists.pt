import torch
from torch.distributions import Distribution, TransformedDistribution, AffineTransform


class Stretched(TransformedDistribution):
    """
    This stretches a distribution via an affine transformation.
    """
    
    def __init__(self, base: Distribution, lower=-0.1, upper=1.1):
        assert lower < 0. and upper > 1., "You need to specify lower < 0 and upper > 1"
        super(Stretched, self).__init__(
            base,
            AffineTransform(loc=lower, scale=upper - lower)
        )
        self.lower = lower
        self.upper = upper
        self.loc = lower
        self.scale = upper - lower


