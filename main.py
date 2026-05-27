from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, AuthenticationError
from dotenv import load_dotenv
from os import getenv
import asyncio
import json
import random
import streamlit as st
import sys


# ===== CUSTOM EXCEPTIONS =====

class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""
    pass


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute."""
    pass


class AgentError(Exception):
    """Raised when the agent encounters an error."""
    pass


# ===== CONFIGURATION =====

load_dotenv()
set_tracing_disabled(disabled=True)


def validate_config() -> tuple[str, str]:
    """Validate and return configuration values."""
    api_key = getenv("GEMINI_API_KEY")
    base_url = getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY is not set in .env file")
    if not base_url:
        raise ConfigurationError("GEMINI_BASE_URL is not set in .env file")
    if len(api_key) < 10:
        raise ConfigurationError("GEMINI_API_KEY appears to be invalid (too short)")

    return api_key, base_url


def initialize_client() -> AsyncOpenAI:
    """Initialize and return the OpenAI client with error handling."""
    try:
        api_key, base_url = validate_config()
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Failed to initialize client: {str(e)}")


# Initialize client and model
try:
    client = initialize_client()
    agent_model = OpenAIChatCompletionsModel(
        openai_client=client,
        model="gemini-2.5-flash"
    )
except ConfigurationError as e:
    client = None
    agent_model = None
    CONFIG_ERROR = str(e)
else:
    CONFIG_ERROR = None


# ===== FUNCTION TOOLS =====

@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.
    """
    try:
        if not city or not city.strip():
            raise ToolExecutionError("City name cannot be empty")

        city = city.strip()
        weather_data = {
            "city": city,
            "temperature": 22,
            "unit": "celsius",
            "condition": "Sunny",
            "humidity": 45,
            "status": "success"
        }
        return json.dumps(weather_data)

    except ToolExecutionError:
        raise
    except Exception as e:
        error_data = {
            "status": "error",
            "error": f"Failed to get weather: {str(e)}",
            "city": city
        }
        return json.dumps(error_data)


@function_tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a mathematical calculation.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide).
        a: The first number.
        b: The second number.
    """
    try:
        if not operation:
            raise ToolExecutionError("Operation cannot be empty")

        operation = operation.lower().strip()
        valid_operations = ["add", "subtract", "multiply", "divide"]

        if operation not in valid_operations:
            raise ToolExecutionError(
                f"Invalid operation '{operation}'. Valid operations: {', '.join(valid_operations)}"
            )

        if operation == "divide" and b == 0:
            raise ToolExecutionError("Division by zero is not allowed")

        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b
        }

        result_data = {
            "operation": operation,
            "a": a,
            "b": b,
            "result": operations[operation],
            "status": "success"
        }
        return json.dumps(result_data)

    except ToolExecutionError as e:
        error_data = {
            "status": "error",
            "error": str(e),
            "operation": operation,
            "a": a,
            "b": b
        }
        return json.dumps(error_data)
    except Exception as e:
        error_data = {
            "status": "error",
            "error": f"Calculation failed: {str(e)}"
        }
        return json.dumps(error_data)


@function_tool
def search_info(query: str) -> str:
    """Search for information on a given topic.

    Args:
        query: The search query or topic.
    """
    try:
        if not query or not query.strip():
            raise ToolExecutionError("Search query cannot be empty")

        query = query.strip()
        result_data = {
            "query": query,
            "results": [
                {"title": f"Information about {query}", "snippet": f"This is detailed information about {query}."},
                {"title": f"{query} - Overview", "snippet": f"A comprehensive overview of {query}."}
            ],
            "total_results": 2,
            "status": "success"
        }
        return json.dumps(result_data)

    except ToolExecutionError as e:
        error_data = {
            "status": "error",
            "error": str(e),
            "query": query
        }
        return json.dumps(error_data)
    except Exception as e:
        error_data = {
            "status": "error",
            "error": f"Search failed: {str(e)}"
        }
        return json.dumps(error_data)


# ===== HELPER FUNCTIONS =====

def format_result(result) -> dict:
    """Format the agent result as both JSON and string."""
    try:
        output_text = result.final_output

        if output_text is None:
            return {
                "json": {"status": "error", "error": "No response from agent"},
                "string": "Error: No response received from the agent.",
                "success": False
            }

        try:
            output_json = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            output_json = {"response": output_text}

        return {
            "json": output_json,
            "string": str(output_text),
            "success": True
        }

    except Exception as e:
        return {
            "json": {"status": "error", "error": str(e)},
            "string": f"Error formatting result: {str(e)}",
            "success": False
        }


def create_agent() -> Agent:
    """Create and return the main agent."""
    if agent_model is None:
        raise AgentError(f"Agent model not initialized: {CONFIG_ERROR}")

    try:
        return Agent(
            name="Main Agent",
            instructions="""You are a helpful assistant that can perform various tasks.
You have access to the following tools:
- get_weather: Get weather information for any city
- calculate: Perform math operations (add, subtract, multiply, divide)
- search_info: Search for information on any topic

Use these tools when appropriate to help answer user questions.
If a tool returns an error, explain the error to the user clearly.""",
            model=agent_model,
            tools=[get_weather, calculate, search_info],
        )
    except Exception as e:
        raise AgentError(f"Failed to create agent: {str(e)}")


async def run_agent(query: str) -> dict:
    """Run the agent with a query and return formatted results."""
    if not query or not query.strip():
        return {
            "json": {"status": "error", "error": "Query cannot be empty"},
            "string": "Please enter a valid question or command.",
            "success": False
        }

    try:
        agent = create_agent()
        result = await Runner.run(starting_agent=agent, input=query.strip())
        return format_result(result)

    except AgentError as e:
        return {
            "json": {"status": "error", "error": str(e)},
            "string": f"Agent Error: {str(e)}",
            "success": False
        }
    except AuthenticationError as e:
        return {
            "json": {"status": "error", "error": "Authentication failed"},
            "string": "Authentication Error: Invalid API key. Please check your GEMINI_API_KEY.",
            "success": False
        }
    except APIConnectionError as e:
        return {
            "json": {"status": "error", "error": "Connection failed"},
            "string": "Connection Error: Unable to reach the API. Please check your internet connection.",
            "success": False
        }
    except APIStatusError as e:
        return {
            "json": {"status": "error", "error": f"API error: {e.status_code}"},
            "string": f"API Error ({e.status_code}): {e.message}",
            "success": False
        }
    except asyncio.TimeoutError:
        return {
            "json": {"status": "error", "error": "Request timed out"},
            "string": "Timeout Error: The request took too long. Please try again.",
            "success": False
        }
    except Exception as e:
        return {
            "json": {"status": "error", "error": str(e)},
            "string": f"Unexpected Error: {str(e)}",
            "success": False
        }


# ===== CUSTOM CSS STYLING =====

def apply_custom_css():
    """Apply custom CSS for stunning UI."""
    st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Tool Cards */
    .tool-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .tool-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }

    .tool-card.weather {
        border-left-color: #f59e0b;
    }

    .tool-card.calculator {
        border-left-color: #10b981;
    }

    .tool-card.search {
        border-left-color: #3b82f6;
    }

    .tool-card h4 {
        margin: 0 0 0.5rem 0;
        font-weight: 600;
        color: #1f2937;
    }

    .tool-card p {
        margin: 0;
        color: #6b7280;
        font-size: 0.9rem;
    }

    .tool-card code {
        background: #f3f4f6;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        display: block;
        margin-top: 0.5rem;
        color: #374151;
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-online {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }

    .status-offline {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    }

    /* Chat Input */
    .stChatInput {
        border-radius: 25px !important;
    }

    .stChatInput > div {
        border-radius: 25px !important;
        border: 2px solid #e5e7eb !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    .stChatInput > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 4px 25px rgba(102, 126, 234, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    /* JSON Display */
    .stJson {
        background: #1e1b4b !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Error/Warning/Success boxes */
    .stAlert {
        border-radius: 12px !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    .animate-pulse {
        animation: pulse 2s infinite;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header animate-fade-in">
        <h1>🤖 AI Agent Studio</h1>
        <p>Your intelligent assistant powered by advanced AI with tool-calling capabilities</p>
    </div>
    """, unsafe_allow_html=True)


def render_tool_card(icon: str, name: str, description: str, example: str, card_class: str):
    """Render a tool card."""
    st.markdown(f"""
    <div class="tool-card {card_class}">
        <h4>{icon} {name}</h4>
        <p>{description}</p>
        <code>💬 "{example}"</code>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(is_online: bool):
    """Render the status badge."""
    if is_online:
        st.markdown("""
        <div class="status-badge status-online">
            <span class="animate-pulse">●</span> System Online
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-badge status-offline">
            <span>●</span> System Offline
        </div>
        """, unsafe_allow_html=True)


# ===== STREAMLIT UI =====

def display_error_banner(error_message: str):
    """Display a prominent error banner."""
    st.error(f"⚠️ {error_message}")


def display_status_indicator():
    """Display the current system status in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        render_status_badge(CONFIG_ERROR is None)
        if CONFIG_ERROR:
            st.caption(f"❌ {CONFIG_ERROR}")
        else:
            st.caption("✨ Powered by Gemini 2.5 Flash")


def main():
    # Page config
    try:
        st.set_page_config(
            page_title="AI Agent Studio",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        pass  # Page config already set

    # Apply custom CSS
    apply_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Tool Suite")
        st.markdown("---")

        render_tool_card(
            "🌤️", "Weather",
            "Get real-time weather information for any city worldwide",
            "What's the weather in Tokyo?",
            "weather"
        )

        render_tool_card(
            "🧮", "Calculator",
            "Perform mathematical operations with precision",
            "Calculate 125 multiplied by 48",
            "calculator"
        )

        render_tool_card(
            "🔍", "Search",
            "Search and retrieve information on any topic",
            "Search for quantum computing",
            "search"
        )

        display_status_indicator()

        # Sidebar footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; opacity: 0.8;">
            <p style="font-size: 0.8rem; color: rgba(255,255,255,0.7);">
                Built with ❤️ using<br/>
                <strong>OpenAI Agents SDK</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    render_header()

    # Check for configuration errors
    if CONFIG_ERROR:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
            <h3 style="color: #991b1b; margin: 0 0 0.5rem 0;">⚠️ Configuration Error</h3>
            <p style="color: #7f1d1d; margin: 0;">{}</p>
        </div>
        """.format(CONFIG_ERROR), unsafe_allow_html=True)

        st.info("👆 Please check your `.env` file and ensure the following variables are set correctly:")
        st.code("""
# .env file should contain:
GEMINI_API_KEY=your_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
        """, language="bash")
        return

    # Quick action buttons
    st.markdown("#### 💡 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🌤️ Check Weather", use_container_width=True):
            st.session_state.quick_prompt = "What's the weather like in New York?"

    with col2:
        if st.button("🧮 Do Math", use_container_width=True):
            st.session_state.quick_prompt = "Calculate 256 divided by 8"

    with col3:
        if st.button("🔍 Search Topic", use_container_width=True):
            st.session_state.quick_prompt = "Search for artificial intelligence"

    with col4:
        if st.button("🎲 Surprise Me", use_container_width=True):
            prompts = [
                "What's the weather in Paris?",
                "Calculate 99 times 101",
                "Search for space exploration",
                "What's the weather in Sydney?",
                "Calculate 1000 minus 777"
            ]
            st.session_state.quick_prompt = random.choice(prompts)

    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize error count for rate limiting feedback
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

    # Handle quick prompts
    quick_prompt = st.session_state.pop("quick_prompt", None)

    # Display chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                if message.get("is_error"):
                    st.error(message["content"])
                else:
                    st.markdown(message["content"])

                if "json_output" in message:
                    with st.expander("📊 View JSON Response"):
                        st.json(message["json_output"])

    # Chat input
    prompt = quick_prompt or st.chat_input("✨ Ask me anything... I can check weather, calculate, or search!")

    if prompt:
        # Validate input
        if len(prompt) > 5000:
            st.warning("⚠️ Your message is too long. Please keep it under 5000 characters.")
            return

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔮 Processing your request..."):
                try:
                    result = asyncio.run(run_agent(prompt))

                    if result.get("success", True):
                        # Reset error count on success
                        st.session_state.error_count = 0

                        # Display string output with nice formatting
                        st.markdown(result["string"])

                        # Display JSON output in expander
                        with st.expander("📊 View JSON Response"):
                            st.json(result["json"])

                        # Add to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["string"],
                            "json_output": result["json"],
                            "is_error": False
                        })
                    else:
                        # Handle known errors from run_agent
                        st.session_state.error_count += 1
                        st.error(result["string"])

                        with st.expander("🔍 Error Details"):
                            st.json(result["json"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["string"],
                            "json_output": result["json"],
                            "is_error": True
                        })

                        # Show helpful tips after multiple errors
                        if st.session_state.error_count >= 3:
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                                        padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                                <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">💡 Troubleshooting Tips</h4>
                                <ul style="color: #1e3a8a; margin: 0; padding-left: 1.5rem;">
                                    <li>Check your internet connection</li>
                                    <li>Verify your API key is correct</li>
                                    <li>Try a simpler question</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                except asyncio.CancelledError:
                    error_msg = "Request was cancelled. Please try again."
                    st.warning(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "is_error": True
                    })

                except RuntimeError as e:
                    if "event loop" in str(e).lower():
                        error_msg = "Session error. Please refresh the page."
                        st.error(error_msg)
                    else:
                        error_msg = f"Runtime Error: {str(e)}"
                        st.error(error_msg)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "is_error": True
                    })

                except Exception as e:
                    st.session_state.error_count += 1
                    error_msg = f"Unexpected Error: {str(e)}"
                    st.error(error_msg)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "is_error": True
                    })

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.error_count = 0
                st.rerun()

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem;">
            <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">
                Made with 🤖 <strong>OpenAI Agents SDK</strong> + ⚡ <strong>Gemini 2.5 Flash</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
        <div style="text-align: right; padding: 0.5rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 0.3rem 0.8rem; border-radius: 20px;
                         font-size: 0.8rem;">
                💬 {message_count} messages
            </span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        st.error(f"Fatal Error: {str(e)}")
        st.info("Please refresh the page or restart the application.")