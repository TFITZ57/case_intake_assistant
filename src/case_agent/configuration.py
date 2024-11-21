from dataclasses import dataclass, field, fields
from firebase_admin import firestore, credentials, initialize_app
from google.cloud import firestore
from langgraph.store.base import BaseStore
from typing import Any, Optional
import logging
import uuid
import os
from langchain_core.runnables import RunnableConfig
from langchain_google_firestore import FirestoreSaver
from typing_extensions import Annotated
from case_agent import prompts

logger = logging.getLogger(__name__)



@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph system."""

    user_id: str = field(default_factory=lambda: str(uuid.uuid4()), metadata={"description": "The ID of the user to remember in the conversation."})
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    case_manager_prompt: str = prompts.CASE_MANAGER_SYSTEM_PROMPT
    trustcall_instruction: str = prompts.TRUSTCALL_INSTRUCTION
    disclaimer: str = prompts.DISCLAIMER

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        conf = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), conf.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})



