# Extensions to torch distributions

Examples:

```python
import torch.distributions as torchd
import probabll.distributions as probd
probd.Kumaraswamy(a=torch.ones(1), b=torch.ones(1))
probd.MixtureOfGaussians(logits=torch.ones(2), components=torchd.Normal(loc=torch.zeros(2), scale=torch.ones(2)))
```

# Installation

* Dependencies

```bash
pip install -r requirements.txt
```

* For developers

```bash
python setup.py develop
```

* For users

```bash
pip install .
```

# Sparse relaxations to binary random variables

Sparse distributions for pytorch:

* HardKumaraswamy
* Mixture of delta and Exponential


# Contributors

Though the history of commits won't necessarily show it, this code base received contributions from:

* [Nicola De Cao](https://github.com/nicola-decao)
* [Jasmijn Bastings](https://github.com/bastings)
* [Bryan Eikema](https://github.com/roxot)
* [Wilker Aziz](https://github.com/wilkeraziz)

# Citation

If you use HardKumaraswamy please cite this paper:

* [Jasmijn Bastings, Wilker Aziz and Ivan Titov. In ACL, 2019.](https://www.aclweb.org/anthology/P19-1284)
