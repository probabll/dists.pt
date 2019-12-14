from setuptools import setup, find_packages

setup(name='probabll.distributions',
      version='1.0',
      description='Extensions to torch.distributions',
      author='Probabll',
      author_email='w.aziz@uva.nl',
      url='https://github.com/probabll/dists.pt',
      packages=find_packages(),
      python_requires='>=3.6',
      include_package_data=True
)
