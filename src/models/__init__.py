"""Model definitions for HCAS-GAN."""

from .deepgaze_wrapper import DeepGazeWrapper
from .discriminator import PatchDiscriminator
from .generator import GeneratorUNet

__all__ = [
	"DeepGazeWrapper",
	"GeneratorUNet",
	"PatchDiscriminator",
]
