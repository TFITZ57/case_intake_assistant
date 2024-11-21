from dataclasses import dataclass, field, fields
from firebase_admin import firestore, credentials, initialize_app
from google.cloud import firestore
from langgraph.store.base import BaseStore
from typing import Any, Optional, Dict, List
import logging
import uuid
import os
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from memory_agent import prompts

logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """Memory class for storing data with metadata."""
    key: str
    value: Dict[str, Any]
    tool_name: str

# Initialize Firebase
cred_path = os.environ.get("FIREBASE_CREDENTIALS")
if not cred_path:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not set")
    
creds = credentials.Certificate(cred_path)
firebase_app = initialize_app(creds)
db = firestore.client()

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

class FireStore(BaseStore):
    def __init__(self, db: Any):
        self.db = db
        self._batch = None

    async def get(self, namespace: tuple[str, str]) -> list[Memory]:
        """Get data from Firestore."""
        doc_ref = self.db.collection(namespace[0]).document(namespace[1])
        doc = doc_ref.get()  # This is synchronous but fast enough for our use case
        return [Memory(key=namespace[0], value=doc.to_dict(), tool_name=namespace[0])] if doc.exists else []

    async def set(self, namespace: tuple[str, str], memory: Memory) -> None:
        """Set data in Firestore."""
        doc_ref = self.db.collection(namespace[0]).document(namespace[1])
        doc_ref.set(memory.value)  # Synchronous operation
        return None

    async def delete(self, namespace: tuple[str, str]) -> None:
        """Delete data from Firestore."""
        doc_ref = self.db.collection(namespace[0]).document(namespace[1])
        doc_ref.delete()  # Synchronous operation
        return None

    async def batch(self) -> None:
        """Start a new batch operation."""
        self._batch = self.db.batch()

    async def abatch(self) -> None:
        """Start a new batch operation."""
        await self.batch()

    def put(self, namespace: tuple[str, str], key: str, value: Dict[str, Any]) -> None:
        """Put a value into the batch."""
        if self._batch is None:
            raise RuntimeError("No batch operation in progress")
        doc_ref = self.db.collection(namespace[0]).document(key)
        self._batch.set(doc_ref, value)

    async def commit(self) -> None:
        """Commit the current batch operation."""
        if self._batch is not None:
            self._batch.commit()  # Synchronous operation
            self._batch = None

db = FireStore(db)