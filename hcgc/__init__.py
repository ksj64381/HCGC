"""
hcgc -- Heterogeneous Graph Coarsening via coalition games.

Public API:
    compress(data, ratio=0.1, ...)  -> HCGCResult
    HCGCResult.data      # compressed HeteroData
    HCGCResult.ratio     # actual achieved compression ratio
    HCGCResult.node_map  # original -> supernode mapping
    HCGCResult.info      # detailed stats
"""

from hcgc._api import compress, HCGCResult

__all__ = ['compress', 'HCGCResult']
__version__ = '0.1.0'
