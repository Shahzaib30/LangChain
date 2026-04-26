import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv("DB_FOR_TOOLS")

def init_db():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS users;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL
                );
            """)
            conn.commit()

@tool()
def add_user(name: str, email: str) -> str:
    "Add a new user to the PostgreSQL database"
    try: 
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id;", (name, email))
                conn.commit()
        return f"User added: {name}"
    except Exception as e:
        return f"Error adding user: {str(e)}"
    
@tool()
def get_users():
    "Retrieve all users from the PostgreSQL database"
    try: 
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor(cursor_factory = RealDictCursor) as cur:
                cur.execute("SELECT id, name, email FROM users;")
                users = cur.fetchall()
        result = {
            "count": len(users),
            "users": users
        }
        return f"""
        Total Users: {len(users)}

        Users:
        {chr(10).join([f"- {u['name']} ({u['email']})" for u in users])}
        """
    except Exception as e:
        return f"Error retrieving users: {str(e)}"
    

tools = [add_user, get_users]
tool_node = ToolNode(tools=tools)
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    
    if not response.tool_calls and "{" in response.content and '"name":' in response.content:
        try:
            start = response.content.find("{")
            end = response.content.rfind("}") + 1
            json_str = response.content[start:end]
            data = json.loads(json_str)
            
            if "name" in data and "arguments" in data:
                response.tool_calls = [{
                    "name": data["name"],
                    "args": data["arguments"],
                    "id": f"call_{uuid.uuid4().hex}"
                }]
                response.content = "" 
        except Exception as e:
            pass 

    return {"messages": [response]}

def should_continue(state: State):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(State)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue , {
    "tools": "tools",
     END: END
})
builder.add_edge("tools", "agent")

graph = builder.compile()


def main():
    init_db()
    system_msg = SystemMessage(content="""
    You are a database assistant.
    Rules:
    1. Always use tools for DB operations.
    2. NEVER modify tool output.
    3. After tool execution, present results in clean natural language.
    4. Never output JSON unless user explicitly asks.
    """)

    state = {"messages": [system_msg]}
    print("Graph initialized. Starting execution...")

    while True:
        user_input = input("\nUser: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        state["messages"].append(HumanMessage(content=user_input))

        result = graph.invoke({"messages": state["messages"]})

        state = result

        print("\n--- FULL OUTPUT ---")
        final_message = state["messages"][-1]
        if isinstance(final_message, AIMessage) and not final_message.tool_calls:
            print("AI:", final_message.content)
        print("--- END OF OUTPUT ---")

if __name__ == "__main__":
    main()
