"""InteractionBench: a real-time interaction benchmark for streaming video systems.

Public API (everything else is implementation detail):

    from interactionbench import (
        load_benchmark, iter_items,          # dataset
        build_model, register_model, register_adapter, ChatModel,   # systems under test
        make_judge, register_judge,          # content judges
        get_protocol, register_protocol, Protocol,                  # test methods
        MetricConfig, score_item, aggregate, # scoring
    )
"""

__version__ = "1.0.0"

from .data import BenchItem, BenchVideo, GTAnswer, iter_items, load_benchmark  # noqa: F401
from .metrics import MetricConfig, aggregate, score_item, summarize  # noqa: F401
from .models import (ChatModel, Generation, build_model, list_models,  # noqa: F401
                     register_adapter, register_model)
from .judges import list_judges, make_judge, register_judge  # noqa: F401
from .protocols import (Protocol, ProtocolConfig, get_protocol,  # noqa: F401
                        list_protocols, register_protocol)
