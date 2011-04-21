from distutils.core import setup

setup(name='neuro_kl',
      version='0.1',
      description='Bayesian estimation of KL and entropy of neural data distributions',
      author='Pietro Berkes',
      author_email='pietro.berkes@googlemail.com ',
      url='https://github.com/pberkes/neuro_kl',
      license='../LICENSE.txt',
      long_description=open('../README.rst').read(),
      packages=['neuro_kl']
      )
