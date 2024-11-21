from langchain_core.messages import merge_message_runs, AIMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from trustcall import create_extractor
from case_agent import configuration, utils
from case_agent.state import State, CaseData, get_schema_json
from firebase_admin import firestore
from datetime import datetime
import json
import logging
import uuid
from functools import partial
import os
logger = logging.getLogger(__name__)

db = firestore.Client(project=os.environ.get("FIRESTORE_PROJECT_ID"))

CaseData = CaseData()

# Initialize the llm
llm = init_chat_model()

# Initialize the Trustcall extractor
trustcall = create_extractor(llm, tools=[CaseData])

# Case manager node
async def case_manager(state: State, config: RunnableConfig) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    conf = configuration.Configuration.from_runnable_config(config)
    user_id = conf.user_id

    # Retrieve existing case data from Firestore
    case_ref = db.collection('cases').document(user_id)
    case_docs = case_ref.get()
    if case_docs.exists:
        case_data = json.dumps(case_docs.to_dict(), indent=2)
    else:
        # Initialize new case data with schema
        case_data = json.dumps(get_schema_json(), indent=2)
        case_ref.set({})

    # Prepare the system prompt with user memories and current time
    prompt = conf.case_manager_prompt.format(
        CaseData=CaseData,
        case_data=case_data,
        disclaimer=conf.disclaimer,
        time=datetime.now().isoformat()
    )

    # Invoke the language model with the prepared prompt and tools
    next_question = await llm.bind_tools([CaseData]).ainvoke(
        [SystemMessage(content=prompt), *state.messages],
        {"configurable": utils.split_model_and_provider(conf.model)},
    )
    return {"messages": [next_question]}

async def extract_data(state: State, config: RunnableConfig, store: BaseStore) -> dict:
    """Extract and store case data from conversation."""
    conf = configuration.Configuration.from_runnable_config(config)
    user_id = conf.user_id

    # Retrieve and format existing case data from Firestore
    case_ref = db.collection('cases').document(user_id)
    case_docs = case_ref.get()
    case_data = [(k, v) for k, v in case_docs.to_dict().items()] if case_docs.exists else None

    # Prepare and invoke extractor
    trustcall_instruction_prompt = conf.trustcall_instruction.format(
        time=datetime.now().isoformat()
    )
    
    # Get messages up to but not including the last tool call
    msgs = state.messages[:-1]
    updated_messages = list(merge_message_runs(messages=[
            SystemMessage(content=trustcall_instruction_prompt),
            *msgs
        ]
    ))
    
    result = await trustcall.ainvoke({
        'messages': updated_messages, 
        'existing_data': case_data if case_data else None
    })

    # Save extracted data
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        doc_id = rmeta.get("json_doc_id", str(uuid.uuid4()))
        doc_ref = db.collection('cases').document(user_id).collection('case_data').document(doc_id)
        await doc_ref.set(r.model_dump(mode="json"))

    # Return tool message to confirm data was stored
    return {
        "messages": [
            ToolMessage(
                content='Case data extracted and stored',
                tool_call_id=str(uuid.uuid4())
            )
        ]
    }

def check_completion(state: State):
    """Check if the interview is complete."""
    msg = state.messages[-1]
    
    # For user messages, always go to extract_data first
    if not isinstance(msg, (AIMessage, ToolMessage)):
        return "extract_data"
        
    # Check for exit conditions
    if isinstance(msg, AIMessage) and "interview is complete" in msg.content.lower():
        return END
        
    # If AI message without tool calls, end cycle to wait for user input
    if isinstance(msg, AIMessage):
        return END
        
    # After extract_data, continue to case_manager
    if isinstance(msg, ToolMessage):
        return "case_manager"
        
    return "case_manager"

# Create the graph
builder = StateGraph(State, config_schema=configuration.Configuration)

# Add nodes and define the flow
builder.add_node("case_manager", case_manager)
builder.add_node("extract_data", extract_data)

# Set the entry point to case manager
builder.add_edge("__start__", "case_manager")

# Add conditional edges from case_manager
builder.add_conditional_edges(
    "case_manager",
    check_completion,
    ["extract_data", END]
)

# Always go to case_manager after updating data
builder.add_edge("extract_data", "case_manager")

graph = builder.compile()
graph.name = "CaseManagerAgent"

__all__ = ["graph"]
