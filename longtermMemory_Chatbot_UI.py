import streamlit as st
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import StateGraph, START, MessagesState
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
import os
import uuid



# Long Term Memory DB
load_dotenv()
db_url = os.getenv("db_url")
pool = ConnectionPool(conninfo=db_url, max_size=5, kwargs={"autocommit": True})

# UI Config

st.set_page_config(page_title="Long Term Memory Chatbot", page_icon=":robot_face:")

if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default_thread"


# Backend

@st.cache_resource
def backend():
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.7, streaming=True)
    
    def chat_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
        # Let LangGraph handle token streaming events from the model.
        response = llm.invoke(state["messages"], config=config)
        return {
            "messages" : [response]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")

    return builder

builder = backend()



# UI

def create_new_chat():
    st.session_state.current_thread = str(uuid.uuid4())

st.sidebar.title("Agent Controls")
st.sidebar.write("Change the thread ID to simulate talking to different clients")
    
st.sidebar.button("➕ New Chat", on_click=create_new_chat)

thread_id = st.sidebar.text_input(
    "Active Thread ID", 
    key="current_thread" 
)

config = {
    "configurable": {
        "thread_id": st.session_state.current_thread
    }
}

st.title("LangGraph Long Term Memory Chatbot")
st.write("If you refresh the page, the chatbot will remember the previous conversation. Change the thread ID to start a new conversation.")

with pool.connection() as conn:
    memory = PostgresSaver(conn)
    memory.setup()

    graph = builder.compile(checkpointer = memory)
    current_state = graph.get_state(config)

    if "messages" in current_state.values:
        for msg in current_state.values["messages"]:
            if msg.type == "human":
                st.chat_message("user").write(msg.content)
            elif msg.type == "ai":
                st.chat_message("assistant").write(msg.content)
    
    if user_input := st.chat_input("Your message"):

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            def stream_response():
                input_state = {
                    "messages": [("user", user_input)]
                }
                for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                    if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
                        if isinstance(chunk.content, str):
                            yield chunk.content
            st.write_stream(stream_response())

