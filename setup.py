from setuptools import setup, find_packages

setup(
    name="lrip",
    version="1.0.0",
    description="Learning Rate Informed Priors for Neural Network Calibration",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.8",
    install_requires=[
        # install torch and torchvision for your CUDA version first
        "torch>=1.10",
        "torchvision>=0.11",
        "hydra-core==1.1.2",          # configs rely on 1.1 behaviour (run dir = cwd)
        "albumentations==1.1.0",      # PadIfNeeded(value=..., mask_value=...) API
        "opencv-python>=4.5",
        "segmentation-models-pytorch>=0.2",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "plotly",
        "terminaltables",
        "wandb",
    ],
    extras_require={
        # 20 Newsgroups preprocessing (tokenizer) as in MbLS
        "nlp": ["tensorflow"],
        "test": ["pytest"],
    },
)
