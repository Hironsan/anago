from setuptools import setup


setup(
    name="anago",
    version="0.1",
    description="Sequence labeling library using Keras",
    keywords=["machine learning", "natural language processing", "sequence labeling",
              "named-entity recognition", "pos tagging", "semantic role labeling"],
    author="Hironsan",
    author_email="hiroki.nakayama.py@gmail.com",
    license="MIT",
    packages=[
        "anago",
        ],
    url="https://github.com/Hironsan/anago",
    install_requires=[
        "six>=1.10.0"
    ],
    package_data={
        '': ['*.csv'],
    }
)