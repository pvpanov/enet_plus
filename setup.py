from setuptools import setup

setup(
    name='enet_plus',
    version='0.1',
    packages=['ridge_plus'],
    url='https://github.com/pvpanov/enet_plus',
    license='MIT',
    author='ppv',
    author_email='pvpanov93@gmail.com',
    description='Elastic net that works simlilar to `sklearn.linear_model.ElasticNet` with - the ability to bound the learned coefficients; - finer control over what coefficients get penalized; - the ability to add extra penalties The package essentially runs `scipy.optimize(method=\'lbfgs\', ...)` in the background.'
)
