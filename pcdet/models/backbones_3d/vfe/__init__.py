from .mean_vfe import MeanVFE
# from .pillar_vfe import PillarVFE
from .pillar_vfe_range_image import PillarVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE
}
