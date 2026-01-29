from setuptools import setup, find_packages

setup(
    name="VoiceBridge",
    version="1.0",
    packages=find_packages(), 
    install_requires=[
        "numpy",
        "sounddevice",
        "webrtcvad",
    ],
    python_requires=">=3.8",
)