import asyncio
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents import AuthorRole
from semantic_kernel.functions import kernel_function

import trafilatura
from duckduckgo_search import DDGS
import json
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# import logging
# logging.basicConfig(level=logging.INFO)



def SaveToJson(data: dict, name: str):

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    filename = f"files/{name}_{timestamp}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"--- {filename} saved!")

    return filename


# Define a sample plugin for the sample
class WebPlugin:


    @kernel_function(description="Use this tool to access web search")
    def get_sources(
        self, query: Annotated[str, "The query to search news information for."]
    ) -> Annotated[str, "Returns Search results information as a JSON string."]:
        print(f'Using tool: ddgsc_search for query:  {query}')


        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=10)

                urls = []
                for result in results:

                    print(result["href"])
                    urls.append(result["href"])
                    
                    
            search_json =  json.dumps({"query": query, 'urls': urls})
            return search_json
        except Exception as e:
            print(f'Error: {str(e)}')
            return str(e)



    @kernel_function(description="Use this tool to extract content from web page")
    def get_content(
        self, url: Annotated[str, "Url to news article to extraact content from"]
    ) -> Annotated[str, "Returns str with content extracted from url"]:
        
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
        }
        try:
            print(f"Extracting content from: {url}")

                # Download the webpage content
            # downloaded = trafilatura.fetch_url(url)
            response = requests.get(url, headers=headers)
            downloaded = response.text
            if not downloaded:
                    print(f"!!!---Failed to download content from: {url}")
                    return f"!!!---Failed to download content from: {url}"

                # Extract the main text content
            text = trafilatura.extract(downloaded)
            if not text:
                    print(f"!!!---No text content extracted from: {url}")
                    return f"!!!---No text content extracted from: {url}"

            print(f"---Successfully extracted content from: {url}")
            return text

        except Exception as e:
                print(f"!!!---Error extracting article content: {str(e)}")
                return str(e)
        

async def main() -> None:
    ai_agent_settings = AzureAIAgentSettings.create(
        model_deployment_name=os.getenv("AGENT_MODEL_DEPLOYMENT_NAME"),  
        project_connection_string=os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"),  
    )

    async with (
        DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(
            credential=creds,
            conn_str=ai_agent_settings.project_connection_string.get_secret_value()) as client,
    ):
    
        # Create agent definition
        agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name="Researcher",
            instructions="""Write an overview of the domain URL. Describe if the content of it is trustworthy or not. 
            #Instructions
            1. Search for the information about the domain of URL, whether it is concideate a trustworthy source or not. Make sure to check the domain name for known missinformation history.
            2. Extract the content of the URL if possible and analyze if it looks like a trustworthy source or not""",
        )

        # Create the AzureAI Agent using the defined client and agent definition
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            plugins=[WebPlugin()],
        )

        # Create a thread to hold the conversation
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread: AzureAIAgentThread | None = None

        user_inputs = [
            {
                "url": "https://suspilne.media/1013583-persij-pontifik-zi-ssa-so-vidomo-pro-roberta-prevo-akij-stav-novim-papou-rimskim/",
                "info": ""
            },
            {
                "url": "http://lenta.ru/",
                "info": ""
            },
            {
                "url": "https://italy.news-pravda.com/en/world/2025/05/08/13577.html",
                "info": ""
            },
            {
                "url": "https://www.la-croix.com/l-inde-a-mene-des-frappes-au-pakistan-intenses-tirs-d-artillerie-au-cachemire-20250506",
                "info": ""
            },
            {
                "url": "https://www.usatoday.com/story/tech/2025/04/22/duolingos-adds-new-chess-course/83209588007/",
                "info": ""
            },
            {
                "url": "https://www.axios.com/local/pittsburgh/2025/04/22/duolingo-chess-learning-app-launch",
                "info": ""
            }
        ]

        try:
            for user_input in user_inputs:
                print()
                print(f"# User: '{user_input['url']}'")
                # Invoke the agent for the specified thread
                response = await agent.get_response(
                    messages=user_input["url"],
                    thread_id=thread,
                )
                print(f"# {response.name}: {response.content}")
                user_input["info"] += response.content.__str__()
                thread = response.thread
            SaveToJson(user_inputs, "domain_agent")
        finally:
            await thread.delete() if thread else None
            await client.agents.delete_agent(agent.id)


if __name__ == "__main__":
    asyncio.run(main())