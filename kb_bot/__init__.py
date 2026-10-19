"""kb_bot: a small, readable RAG question-answering bot over local documents."""

from .bot import Answer, KnowledgeBase
from .config import Settings

__all__ = ["Answer", "KnowledgeBase", "Settings"]
__version__ = "0.2.0"
