import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
import datetime

# --- Hardcoded Gemini API Key ---
GOOGLE_API_KEY = "AIzaSyCqxkfddGd92M7nkCt_RDAlX3Uq56pM9u0"

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
        f"🎨 Top UI/UX Trends for {today} on '{query}':\n"
        "1️⃣ Voice-based UI explorations\n"
        "2️⃣ AI-driven prototyping\n"
        "3️⃣ Inclusive design principles\n"
        "4️⃣ Design systems for multi-brand products\n"
        "5️⃣ Gamified microinteractions in web apps"
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

# --- Streamlit UI Setup ---
st.set_page_config(page_title="💡 UX Trend Agent", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body { background-color: #f7f9fc; }
    .main { background-color: #ffffff; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .stTextInput > div > div > input {
        border: 2px solid #6c63ff;
        border-radius: 10px;
        padding: 0.75rem;
    }
    .stButton > button {
        background: linear-gradient(to right, #6c63ff, #4e54c8);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("""
    <div style='text-align:center;'>
        <h1 style='color:#4e54c8;'>🧠 UX Trend Agent</h1>
        <h4>Stay Ahead with the Latest in Design, AI & User Experience</h4>
    </div>
""", unsafe_allow_html=True)

# --- Input Section ---
with st.container():
    query = st.text_input("🎯 Ask about a UI/UX topic or trend:", placeholder="e.g. What's trending in mobile UX?")

    if st.button("🔍 Discover Now") and query:
        with st.spinner("✨ Fetching the latest insights for you..."):
            try:
                result = agent_executor.run(query)
                st.success("✅ Here's what we found:")
                st.markdown(f"""
                    <div style='background:#f0f2f6; padding:1rem; border-left: 5px solid #6c63ff; border-radius: 10px;'>
                        <pre style='white-space: pre-wrap; word-break: break-word;'>{result}</pre>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"🚫 Error: {e}")
