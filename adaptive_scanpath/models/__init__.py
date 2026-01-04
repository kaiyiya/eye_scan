"""
模型模块初始化
"""

from .sphere_cnn import SphericalCNN, MultiScaleSphericalCNN
from .policy_network import PolicyNetwork, StoppingNetwork, ContextRNN, FeatureUpdater
from .adaptive_scanpath import AdaptiveScanPath

__all__ = [
    'SphericalCNN',
    'MultiScaleSphericalCNN',
    'PolicyNetwork',
    'StoppingNetwork',
    'ContextRNN',
    'FeatureUpdater',
    'AdaptiveScanPath'
]
