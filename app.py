from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, AsyncGenerator
import json
import os
import asyncio
from datetime import datetime
import uuid
from openai import OpenAI
# Import your existing components
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Travel Assistant API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str | None = None

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    messages: List[ChatMessage]

# In-memory session storage (use Redis in production)
sessions: Dict[str, Dict[str, Any]] = {}

# Your existing components
class MemoryState(Dict[str, Any]):
    messages: List[Dict[str, str]]

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.7)
hotel_tool = TavilySearch(k=3, api_key=os.getenv("TAVILY_API_KEY"))
flight_tool = TavilySearch(k=3, api_key=os.getenv("TAVILY_API_KEY"))

def save_session_memory(session_id: str, memory: Dict[str, Any]):
    """Save session memory."""
    sessions[session_id] = memory
    # Optionally save to file for persistence
    try:
        with open(f"session_{session_id}.json", "w") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save session {session_id}: {e}")

def load_session_memory(session_id: str) -> Dict[str, Any]:
    """Load session memory."""
    if session_id in sessions:
        return sessions[session_id]
    
    # Try to load from file
    try:
        if os.path.exists(f"session_{session_id}.json"):
            with open(f"session_{session_id}.json", "r") as f:
                memory = json.load(f)
                sessions[session_id] = memory
                return memory
    except Exception as e:
        print(f"Warning: Failed to load session {session_id}: {e}")
    
    return {"messages": []}

# Your existing agent functions (adapted for async)
def query_coordinator(state: MemoryState):
    last_user_msg = state["messages"][-1]["content"]
    system_prompt = """You are a query coordinator for a travel assistant. 
        Analyze the user's message and decide the appropriate action:
        - If asking about hotels, accommodations,resorts or places to stay: respond 'hotel'
        - If asking about flights, airlines, or air travel: respond 'flight' 
        - If asking about other travel related queries: respond 'other'

        Respond with ONLY one word: 'hotel', 'flight', or 'other'."""

    try:
        decision = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_user_msg},
        ]).content.strip().lower()

        if "hotel" in decision:
            return {"next": "hotel_agent"}
        elif "flight" in decision:
            return {"next": "flight_agent"}
        else:
            return {"next": "coordinator_reply"}
    except Exception as e:
        print(f"Error in coordinator: {e}")
        return {"next": "coordinator_reply"}

def coordinator_reply(state: MemoryState):
    last_user_msg = state["messages"][-1]["content"]
    system_prompt = """You are a helpful travel assistant. The user has asked something that's not specifically about flights or hotels. 
Provide a helpful response and gently guide them towards flight or hotel queries if appropriate.
Keep your response concise and friendly."""

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_user_msg}
        ]).content
    except Exception as e:
        response = f"I'm here to help with flights and hotels. Could you please specify what you're looking for?"

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response}]
    }

def format_results(results: dict) -> str:
    """Helper function to format Tavily search results."""
    if not results or "results" not in results:
        return "No results found."

    formatted = []
    # prompt_sys="""Clean the response you are receiving from Tavily search results."""
    # try:
    #     response = llm.invoke([
    #         {"role": "system", "content": prompt_sys},
    #         {"role": "user", "content": results}
    #     ]).content
    # except Exception as e:
    #     response = "No results found."
    # return response
    for i, r in enumerate(results["results"], 1):
        title = r.get('title', 'N/A')
        url = r.get('url', 'N/A')
        content = r.get('content', 'N/A')
        formatted.append(f"{i}. **{title}**\n\nURL: {url}\n\n{content}")
    
    return "\n\n---\n\n".join(formatted)

def hotel_agent(state: MemoryState):
    query = state["messages"][-1]["content"]
    try:
        results = hotel_tool.invoke({"query": f"hotels {query}"})
        reply = f"Here are some hotel options I found:\n\n{format_results(results)}"
    except Exception as e:
        reply = f"Sorry, I encountered an error while searching for hotels: {str(e)}"

    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}]}

def flight_agent(state: MemoryState):
    query = state["messages"][-1]["content"]
    try:
        results = flight_tool.invoke({"query": f"flights {query}"})
        reply = f"Here are some flight options I found:\n\n{format_results(results)}"
    except Exception as e:
        reply = f"Sorry, I encountered an error while searching for flights: {str(e)}"

    return {"messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# Create graph
def create_graph():
    graph = StateGraph(MemoryState)
    graph.add_node("coordinator", query_coordinator)
    graph.add_node("coordinator_reply", coordinator_reply)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.set_entry_point("coordinator")
    graph.add_conditional_edges(
        "coordinator",
        lambda x: x["next"],
        {
            "hotel_agent": "hotel_agent",
            "flight_agent": "flight_agent",
            "coordinator_reply": "coordinator_reply",
        },
    )
    graph.add_edge("hotel_agent", END)
    graph.add_edge("flight_agent", END)
    graph.add_edge("coordinator_reply", END)
    return graph.compile()

travel_graph = create_graph()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the frontend HTML."""
    try:
        with open("frontend.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found. Please create frontend.html</h1>", status_code=404)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Load session memory
        memory = load_session_memory(session_id)
        
        # Add user message
        user_message = {
            "role": "user", 
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        }
        memory["messages"].append(user_message)
        
        # Process through graph
        final_state = None
        for chunk in travel_graph.stream(memory):
            final_state = chunk
        
        # Extract response
        if final_state:
            for agent in ["flight_agent", "hotel_agent", "coordinator_reply"]:
                if agent in final_state:
                    memory.update(final_state[agent])
                    break
        
        # Save session
        save_session_memory(session_id, memory)
        
        # Format response
        assistant_message = memory["messages"][-1]
        
        return ChatResponse(
            response=assistant_message["content"],
            session_id=session_id,
            messages=[ChatMessage(**msg) for msg in memory["messages"]]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint."""
    
    # Validate input
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    async def generate_response():
        try:
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())
            
            # Load session memory
            memory = load_session_memory(session_id)
            
            # Add user message with timestamp
            user_message = {
                "role": "user", 
                "content": request.message.strip(),
                "timestamp": datetime.now().isoformat()
            }
            memory["messages"].append(user_message)
            
            # Send session info first
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            # Process through graph and stream
            full_response = ""
            final_state = None
            
            try:
                for chunk in travel_graph.stream(memory):
                    final_state = chunk
                    
                    # Extract response from different agents
                    response_content = ""
                    for agent in ["flight_agent", "hotel_agent", "coordinator_reply"]:
                        if agent in chunk and "messages" in chunk[agent]:
                            messages = chunk[agent]["messages"]
                            if messages and len(messages) > 0 and messages[-1].get("role") == "assistant":
                                response_content = messages[-1]["content"]
                                break
                    
                    if response_content and response_content != full_response:
                        # Stream new content
                        new_content = response_content[len(full_response):]
                        for char in new_content:
                            char_data = json.dumps({'type': 'content', 'content': char})
                            yield f"data: {char_data}\n\n"
                            await asyncio.sleep(0.008)  # Small delay for streaming effect
                        full_response = response_content
                
                # Update memory with final state
                if final_state:
                    for agent in ["flight_agent", "hotel_agent", "coordinator_reply"]:
                        if agent in final_state:
                            memory.update(final_state[agent])
                            break
                
                # Ensure we have a response in memory
                if not full_response and memory["messages"] and memory["messages"][-1]["role"] == "user":
                    # Add a fallback response if something went wrong
                    fallback_response = "I apologize, but I'm having trouble processing your request right now. Please try again."
                    memory["messages"].append({
                        "role": "assistant", 
                        "content": fallback_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    for char in fallback_response:
                        char_data = json.dumps({'type': 'content', 'content': char})
                        yield f"data: {char_data}\n\n"
                        await asyncio.sleep(0.008)
                
            except Exception as graph_error:
                error_message = f"I encountered an error while processing your request: {str(graph_error)}"
                memory["messages"].append({
                    "role": "assistant", 
                    "content": error_message,
                    "timestamp": datetime.now().isoformat()
                })
                for char in error_message:
                    char_data = json.dumps({'type': 'content', 'content': char})
                    yield f"data: {char_data}\n\n"
                    await asyncio.sleep(0.008)
            
            # Save session
            save_session_memory(session_id, memory)
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            error_data = json.dumps({'type': 'error', 'error': str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session history."""
    memory = load_session_memory(session_id)
    return {"session_id": session_id, "messages": memory["messages"]}

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear session history."""
    if session_id in sessions:
        del sessions[session_id]
    
    # Remove file if exists
    try:
        if os.path.exists(f"session_{session_id}.json"):
            os.remove(f"session_{session_id}.json")
    except Exception as e:
        print(f"Warning: Failed to remove session file: {e}")
    
    return {"message": "Session cleared successfully"}

@app.options("/api/chat/stream")
async def chat_stream_options():
    """Handle preflight requests for CORS."""
    return {"message": "OK"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)