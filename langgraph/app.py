from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import ToolNode

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from chainlit.types import ThreadDict
import chainlit as cl

model = ChatOpenAI(streaming=True)

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", call_model)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for message, metadata in graph.stream({"messages": [HumanMessage(content=message.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            message.content
            and not isinstance(message, HumanMessage)
            and metadata["langgraph_node"] == "chatbot"
        ):
            await final_answer.stream_token(message.content)

    await final_answer.send()
