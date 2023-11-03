import os
from dotenv import load_dotenv
import streamlit as st
import pprint
import sys


from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from fastapi import FastAPI
import slack_sdk

import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())


load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")


urls=[]

# 1. Tool for search
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    slack_token_paul = "xoxb-6131557495012-6135383978230-HBelSZGQdYaCDh2Zlf0qERbE"  # Make sure to replace this with your actual token.
    client_paul = slack_sdk.WebClient(token=slack_token_paul)

    msg="I'm reading this : "+str(url)
    print(msg)
    client_paul.chat_postMessage(
    channel="C0645TGK2DR", text=f"{msg}")
    
    print("Here scrape_website")
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }
    # Convert Python object to JSON string
    data_json = json.dumps(data)
    
    urls.append(url)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        # print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    print("Here summary")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        print("Here ScrapeWebsiteTool")
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can produce deep dive analyses of any given topic; 
            you do not make things up, you will try as hard as possible to gather facts & data by scraping articles and links !
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ You select a few links and articles and you will scrape it to gather more information, this is important in order to give deep insightful analyses and not just verbose.
            3/ You should not make things up, you should only write facts & data that you have gathered
            4/ Assume you are talking to someone smart. Do deep analyses of the articles you read by scraping them and try to be insightful without doing verbose
            5/ Assume you are talking to someone smart. Do deep analyses of the articles you read by scraping them and try to be insightful without doing verbose"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
#"gpt-3.5-turbo-16k-0613"
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


def paul(res):
    chat = ChatOpenAI(temperature=1, model_name="gpt-4")
    prompt="""
    You are Paul Llagonne, a VC analyst at Daphni (a french fund). 
Keep these very important criteria in mind :
1. A catchy opening that includes an emoji.
2. Style and tone : Professional, Authoritative and direct
3. A concise, value-packed title for the main content.
4. Bullet points that detail key insights, lessons, or data points.
5. Use of industry-specific terminology and concepts.
6. A concluding remark that ties the insights together and reinforces the main point.
7. An invitation for engagement related to the industry or topic discussed.
8. Incorporate an example or a short case study relevant to the main content.
9. Use at least one numbered list to highlight steps or benefits.
10. Include a sign-off with a call to action for readers.

Use it to generate a concise but value-packed LinkedIn post from the following content :\n
    """+res
    history = ChatMessageHistory()


    history.clear()

    history.add_user_message(prompt)

    ai_response = chat(history.messages).content

    return ai_response


# 4. Use streamlit to create a web app
def main(query):
    slack_token_paul = "xoxb-6131557495012-6135383978230-HBelSZGQdYaCDh2Zlf0qERbE"  # Make sure to replace this with your actual token.
    client_paul = slack_sdk.WebClient(token=slack_token_paul)

    result = agent({"input": query})

    final=paul(result['output'])

    client_paul.chat_postMessage(
        channel="C0645TGK2DR", text=f"{final}")
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python app.py <research_goal>")
    else:
        main(sys.argv[1])


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content

# def log_and_print_online(role, content=None):
#     slack_token_paul = "xoxb-6131557495012-6135383978230-HBelSZGQdYaCDh2Zlf0qERbE"  # Make sure to replace this with your actual token.
#     client_paul = slack_sdk.WebClient(token=slack_token_paul)

#     client_paul.chat_postMessage(
#     channel="C0645TGK2DR", text=f"{content}")

