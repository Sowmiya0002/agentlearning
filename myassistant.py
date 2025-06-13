import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent_toolkits import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool as LangChainTool
from langchain_core.runnables import RunnableLambda
import datetime

# --- Hardcoded Gemini API Key ---
GOOGLE_API_KEY = "your-google-api-key"

# --- Set up LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4
)

# --- Tools: Define a fake daily UI/UX trends function ---
def fetch_daily_uiux_trends(_):
    today = datetime.date.today().strftime("%B %d, %Y")
    return (
        f"Here are the top UI/UX trends for {today}:\n"
        "1. Neumorphism revival in dashboard design\n"
        "2. AI-assisted design tools gaining traction\n"
        "3. Increased focus on accessibility and dark mode\n"
        "4. More micro-interactions in mobile UI\n"
        "5. Multi-modal UI exploration with voice+touch"
    )

# --- Tool wrapper for LangChain ---
daily_trend_tool = LangChainTool(
    name="Daily UI/UX Trends",
    func=fetch_daily_uiux_trends,
    description="Fetch today's top 5 UI/UX design trends for designers."
)

# --- Agent Setup ---
tools = [daily_trend_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful UX mentor for designers. Use tools if needed."),
    ("user", "{input}")
])

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Daily UX Mentor", layout="wide")
st.title("🎨 Daily UX Mentor — Powered by LangChain Agent")

user_query = st.text_input("Ask your question about UI/UX trends:", placeholder="e.g. What are today's design trends?")

if user_query:
    with st.spinner("Thinking like a design mentor..."):
        try:
            response = agent_executor.invoke({"input": user_query})
            st.success("Here's your UX insight:")
            st.write(response["output"])
        except Exception as e:
            st.error("❌ Something went wrong: " + str(e))
