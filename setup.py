from setuptools import setup, find_packages

setup(
    name="algebraic_value_editing",
    description=(
        "Tools for testing the algebraic value-editing conjecture (AVEC) on"
        " language models"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        (
            "transformer-lens @"
            " git+https://github.com/neelnanda-io/TransformerLens@9e034eef796ed32b0e473ca9f3c9d31e5d5046e7"
        ),
        "torch==1.13.1",
        "numpy>=1.22.1",
        "pandas>=1.4.4",
        "jaxtyping>=0.2.14",
        "prettytable>=3.6.0",
        "openai>=0.27.2",
        "nltk>=3.8.1",
        "kaleido>=0.2.1",
        "pytest",
        "plotly",
        "nbformat",
        "Ipython",
        "ipywidgets",
        "tuned_lens",
    ],
    extras_require={
        "dev": [
            "pytest",
            "notebook",  # liked by vscode
        ]
    },
)
