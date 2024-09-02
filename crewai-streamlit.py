
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from crewai import Agent, Task, Crew, Process
import yfinance as yf
from datetime import datetime
import os
import json
import streamlit as st


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock


yahoo_finance_tool = Tool(
    name="Tahoo finance tool",
    description="Fetches stocks prices for {ticket} from the last year about a specific company from Tahoo Finance API.",
    func=lambda ticket: fetch_stock_price(ticket)
)

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You're highly experienced in analyzing the price of an specific stock 
    and make predctions about its futuro price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)

getStockPrice = Task(
    description="Analyzing the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output="""Specify the current trend price - up, down ir sideways.
    eg. STOCK = 'APPL, price UP'""",
    agent=stockPriceAnalyst
)

search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company. Specify current trend - up, down or sideways with the news context. For each request stock asset, 
    specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market and news and have tracked assets for more then 10 years.    
    You're also master level analyst in the tradicional markets and have a deep understanding of human psychology.
    You undertanding news, theirs tittles and information, but you look at those with a health dose of skepticism. 
    You consider also the source of the news articles.""",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tool=[search_tool],
    allow_delegation=False
)

getNews = Task(
    description="""Take the Stock and aways include BTC to it (if not request).
    Use the search tool to search each one individually.
    Compose the results into a helpfull report""",
    expected_output="""A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BAED ON NEWS>
    <TREND PREDICTION>
    <FEER/GREED SCORE>""",
    agent=newsAnalyst
)

stockAnalystWriter = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trends price and news and Write an insightfull compelling and informastive 3 paragraph long newsletter based on the stock report and price trend. """,
    backstory="""You're widely accepted as the best stock analyst in the market. You undertand complex concepts and create compelling stories and narratives that resonate 
    with wider audiences.
    You're understand macro factos and combine multiples theories - eg. cycle theory and fundamentals analyses.
    You're able to hold multiples opnions when analyzing anything.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True
)

writeAnalyses = Task(
    description="""Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are near future considerations?
    Include the previous analyses of stock trend and news summary.""",
    expected_output="""An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:
    - 3 bullets execuritve summary
    - Introducion - set the overall picture and spike up the interest
    - Main part provides the meat of the analysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction - up, down or sideways""",
    agent=stockAnalystWriter,
    context=[getStockPrice, getNews]
)

crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, getNews, writeAnalyses],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

# results = crew.kickoff(inputs={'ticket': 'AAPL'})

# results['final_output']

with st.sidebar:
    st.header('Enter the stock to research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results = crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research: ")
        st.write(results['final_output'])
