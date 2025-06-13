import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
import datetime

# --- Hardcoded Gemini API Key ---
GOOGLE_API_KEY = "AIzaSyB5sW_eEBjodfjeORfXnAARBTFEREvIp9k"

# --- Set up LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4
)

# --- Tool: Daily UI/UX Trends ---
def fetch_uiux_trends(query: str) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    return (
        f"Here are the top UI/UX trends for {today} based on '{query}':\n"
        "1. Voice-based UI explorations\n"
        "2. AI-driven prototyping\n"
        "3. Inclusive design principles\n"
        "4. Design systems for multi-brand products\n"
        "5. Gamified microinteractions in web apps"
    )

trend_tool = Tool(
    name="UI/UX Trend Helper",
    func=fetch_uiux_trends,
    description="Helps designers get daily UI/UX trends given a topic or question."
)

# --- Initialize LangChain Agent ---
agent_executor = initialize_agent(
    tools=[trend_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="UX Trend Agent", layout="wide")
st.title("🧠 UX Trend Agent — Stay Updated with LangChain Agents")

query = st.text_input("🎯 Ask about a UI/UX topic or trend:", placeholder="e.g. What's trending in mobile UX?")

if query:
    with st.spinner("Finding the latest trends for you..."):
        try:
            result = agent_executor.run(query)
            st.success("🔍 Trend Insights:")
            st.write(result)
        except Exception as e:
            st.error(f"🚫 Error: {e}")
