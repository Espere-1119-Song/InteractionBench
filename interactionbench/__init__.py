"""InteractionBench: a real-time interaction benchmark for streaming video systems."""

__version__ = "1.0.0"

from .data import BenchItem, BenchVideo, GTAnswer, iter_items, load_benchmark  # noqa: F401
from .metrics import MetricConfig, aggregate, score_item, summarize  # noqa: F401
from .models import (ChatModel, Generation, build_model, list_models,  # noqa: F401
                     register_adapter, register_model)
from .judges import list_judges, make_judge, register_judge  # noqa: F401
from .protocols import (Protocol, ProtocolConfig, get_protocol,  # noqa: F401
                        list_protocols, register_protocol)
