from setuptools import setup, find_namespace_packages

setup(name='probabll.distributions',
      version='1.0',
      description='Extensions to torch.distributions',
      author='Probabll',
      author_email='w.aziz@uva.nl',
      url='https://github.com/probabll/dists.pt',
      packages=find_namespace_packages(include=['probabll.*']),
      python_requires='>=3.6',
      include_package_data=True
)
