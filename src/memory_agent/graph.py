from langchain_core.messages import merge_message_runs, AIMessage, SystemMessage, HumanMessage, ToolMessage
from memory_agent.state import State, CaseData, UserData, get_schema_json
from langchain_core.runnables import RunnableConfig
from memory_agent.configuration import FireStore, Memory, db
from memory_agent import configuration
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from trustcall import create_extractor
from typing import TypedDict, Literal
from memory_agent import prompts
from datetime import datetime
import logging
import uuid
import json
import os

logger = logging.getLogger(__name__)

# Initialize the llm
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Case manager node
async def case_manager(state: State, config: RunnableConfig) -> dict:
    """Manages the case intake interview process by generating contextual questions 
    when case data is missing, extracting relevant information from user responses, 
    and persisting the data to the database backend. The node handles the full 
    interview lifecycle including initial data gathering, follow-up clarifications, 
    and data validation before storage."""
    conf = configuration.Configuration.from_runnable_config(config)
    user_id = conf.user_id

    # Retrieve existing case data from Firestore
    namespace = ('Case', user_id)
    case_data_list = await db.get(namespace)
    
    if case_data_list:
        case_memory = case_data_list[0].value
    else:
        # Initialize new case data with schema
        case_memory = get_schema_json()
        await db.set(namespace, Memory(key=user_id, value=case_memory, tool_name="Case")
                     )
    # Prepare the system prompt with user memories and current time
    prompt = conf.case_manager_prompt.format(
        schema=get_schema_json(),
        case_memory=json.dumps(case_memory, indent=2),
        disclaimer=conf.disclaimer,
        time=datetime.now().isoformat()
    )
    # Filter messages for the conversation
    filtered_messages = [
        msg for msg in state.messages 
        if isinstance(msg, (HumanMessage, AIMessage))
    ]
    # Invoke the language model with the prepared prompt
    next_question = await llm.ainvoke(
        [SystemMessage(content=prompt), *filtered_messages]
    )
    return {"messages": [next_question]}

async def update_case(state: State, config: RunnableConfig) -> dict:
    """Reflect on the chat history and update the memory collection."""
    conf = configuration.Configuration.from_runnable_config(config)
    user_id = conf.user_id

    namespace = ('Case', user_id)
    case_data_list = await db.get(namespace)
    existing_memories = None
    
    if case_data_list:
        existing_memories = [(item.key, "Case", item.value) for item in case_data_list]
    
    # Merge the chat history and the instruction
    trustcall_prompt = prompts.TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[
        SystemMessage(content=trustcall_prompt), 
        *state.messages[:-1]
    ]))

    trustcall = create_extractor(
        llm=llm,
        tools=[CaseData],
        tool_choice="CaseData",
        enable_insert=True
    )

    updated_case_data = await trustcall.ainvoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(updated_case_data["responses"], updated_case_data["response_metadata"]):
        doc_id = rmeta.get("json_doc_id", str(uuid.uuid4()))
        await db.set(
            namespace,
            Memory(key=doc_id, value=r.model_dump(mode="json"), tool_name="Case")
        )

    return {
        "messages": [
            ToolMessage(
                content="Case data updated", 
                tool_call_id=str(uuid.uuid4())
            )
        ]
    }

async def update_user(state: State, config: RunnableConfig, db: FireStore) -> dict:
    """Reflect on the chat history and update the memory collection."""
    conf = configuration.Configuration.from_runnable_config(config)
    user_id = conf.user_id

    namespace = ('User', user_id)
    items = db.get(namespace)
    tool_name = "User"
    existing_memories = ([(item.key, tool_name, item.value)
                        for item in items]
                        if items 
                        else None)
    
    trustcall = create_extractor(
        llm=llm,
        tools=[UserData],
        tool_choice=tool_name,
        enable_insert=True
    )

    # Merge the chat history and the instruction
    trustcall_prompt = prompts.TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[
        SystemMessage(content=trustcall_prompt),
        *state.messages[:-1]
    ]))

    updated_user_data = await trustcall.ainvoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })  
    
    ## Save the memories from Trustcall to the store
    for r, rmeta in zip(updated_user_data["responses"], 
                        updated_user_data["response_metadata"]):
        db.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
        )

    return {"messages": [ToolMessage(content="User data updated", 
                                    tool_call_id=str(uuid.uuid4()))]}

async def end_interview(state: State, config: RunnableConfig) -> dict:
    """Node to properly end the interview."""
    return {"messages": [*state.messages, AIMessage(content="Thank you for your time. The interview is now complete.")]}

async def router_node(state: State) -> str:
    """Route the conversation based on the last message."""
    msg = state.messages[-1]
    
    # Check for termination
    if isinstance(msg, HumanMessage) and msg.content.lower() in ["quit", "exit", "terminate"]:
        return "end_interview"
    
    # Main interview flow
    if msg.additional_kwargs.get("tool_calls"):
        tool_names = [tc["function"]["name"] for tc in msg.additional_kwargs["tool_calls"]]
        if "UserData" in tool_names:
            return "update_user"
        if "CaseData" in tool_names:
            return "update_case"
    return "case_manager"

# Create the graph
builder = StateGraph(State, config_schema=configuration.Configuration)

# Add nodes
builder.add_node("case_manager", case_manager)
builder.add_node("update_case", update_case)
builder.add_node("update_user", update_user)
builder.add_node("end_interview", end_interview)

# Set the entry point
builder.add_edge("__start__", "case_manager")

# Add conditional edges for case_manager
builder.add_conditional_edges(
    "case_manager",
    router_node,
    {
        "update_case": "update_case",
        "update_user": "update_user",
        "end_interview": "end_interview",
        END: END
    }
)

# Add edge from end_interview to END
builder.add_edge("end_interview", END)

# Compile the graph
graph = builder.compile()
graph.name = "CaseManagerAgent"

__all__ = ["graph"]

