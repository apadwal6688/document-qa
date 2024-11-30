import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
import crewai

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["api_keys"]["alpha_vantage"]
openai.api_key = st.secrets["api_keys"]["openai"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage RAG System with Crew AI")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter with Crew AI"]
)

# Initialize Crew AI Agents
def initialize_agents():
    agents = {
        "market_analyst": crewai.Agent(
            name="Market Analyst",
            description="Analyzes market trends and provides insights based on ticker data.",
            tools=["summarize", "analyze"],
        ),
        "company_analyst": crewai.Agent(
            name="Company Analyst",
            description="Examines company-specific news and data to identify key takeaways.",
            tools=["summarize", "filter"],
        ),
        "risk_analyst": crewai.Agent(
            name="Risk Analyst",
            description="Identifies risks in market trends and news sentiment for investors.",
            tools=["analyze", "evaluate"],
        ),
        "news_editor": crewai.Agent(
            name="News Editor",
            description="Compiles and organizes information into a cohesive newsletter.",
            tools=["organize", "compose"],
        ),
    }

    orchestrator = crewai.Orchestrator(
        agents=agents,
        coordinator="CollaborativeCoordinator",
    )
    return orchestrator

# Assign tasks to agents
def assign_tasks(orchestrator, combined_data):
    tasks = [
        {"agent": "market_analyst", "task": f"Analyze market trends:\n{json.dumps(combined_data['tickers'], indent=2)}"},
        {"agent": "company_analyst", "task": f"Summarize company news:\n{json.dumps(combined_data['news'], indent=2)}"},
        {"agent": "risk_analyst", "task": "Evaluate potential risks from the provided data."},
        {"agent": "news_editor", "task": "Compile findings into a newsletter."},
    ]
    results = orchestrator.execute(tasks)
    return results

# Generate Newsletter using Crew AI
def generate_newsletter_with_agents():
    news_collection = client.get_or_create_collection("news_sentiment_data")
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    
    try:
        news_results = news_collection.get()
        news_data = [json.loads(doc) for doc in news_results["documents"]]
        
        ticker_data = {}
        for data_type in ["top_gainers", "top_losers", "most_actively_traded"]:
            results = ticker_collection.get(ids=[data_type])
            ticker_data[data_type] = json.loads(results["documents"][0])
        
        combined_data = {"news": news_data[:10], "tickers": ticker_data}
        
        orchestrator = initialize_agents()
        agent_results = assign_tasks(orchestrator, combined_data)
        
        # Use the multi-agent system outputs for the newsletter
        market_insights = agent_results.get("market_analyst", "No insights available.")
        company_highlights = agent_results.get("company_analyst", "No highlights available.")
        risk_evaluation = agent_results.get("risk_analyst", "No risk evaluation available.")
        newsletter_content = agent_results.get("news_editor", "No newsletter generated.")

        # Generate and display the newsletter
        newsletter = f"""
        ### Alpha Vantage Daily Newsletter
        
        **Market Insights**:
        {market_insights}
        
        **Company Highlights**:
        {company_highlights}
        
        **Risk Evaluation**:
        {risk_evaluation}
        
        **Final Newsletter**:
        {newsletter_content}
        """
        
        st.subheader("Generated Newsletter with Crew AI Agents")
        st.markdown(newsletter)
        
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")

### Function to Load News Data ###
def load_news_data():
    news_collection = client.get_or_create_collection("news_sentiment_data")
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()
        if 'feed' in data:
            news_items = data['feed']
            for i, item in enumerate(news_items, start=1):
                document = {
                    "id": str(i),
                    "title": item["title"],
                    "url": item["url"],
                    "time_published": item["time_published"],
                    "source": item["source"],
                    "summary": item.get("summary", "N/A"),
                    "topics": [topic["topic"] for topic in item.get("topics", [])],
                    "overall_sentiment_label": item.get("overall_sentiment_label", "N/A"),
                    "overall_sentiment_score": item.get("overall_sentiment_score", "N/A"),
                    "ticker_sentiments": [
                        {
                            "ticker": ticker["ticker"],
                            "relevance_score": ticker["relevance_score"],
                            "ticker_sentiment_label": ticker["ticker_sentiment_label"],
                            "ticker_sentiment_score": ticker["ticker_sentiment_score"],
                        }
                        for ticker in item.get("ticker_sentiment", [])
                    ],
                }
                topics_str = ", ".join(document["topics"])
                ticker_sentiments_str = json.dumps(document["ticker_sentiments"])
                news_collection.add(
                    ids=[document["id"]],
                    metadatas=[{
                        "source": document["source"],
                        "time_published": document["time_published"],
                        "topics": topics_str,
                        "overall_sentiment": document["overall_sentiment_label"],
                        "ticker_sentiments": ticker_sentiments_str,
                    }],
                    documents=[json.dumps(document)]
                )
            st.success(f"Inserted {len(news_items)} news items into ChromaDB.")
        else:
            st.error("No news data found.")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Generate Newsletter with Crew AI":
    generate_newsletter_with_agents()
