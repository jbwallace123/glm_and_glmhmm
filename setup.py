from setuptools import find_packages, setup

setup(
    name='sglm',
    packages=find_packages(),
    version='0.0.1',
    description='A GLM Pipeline built using ElasticNet from scikit-learn',
    author='Janet Berrios Wallace',
    author_email='janet_wallace@hms.harvard.edu',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'PyYAML', 'seaborn'],
    license='',
)
