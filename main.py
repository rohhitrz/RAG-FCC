from dotenv import load_dotenv
from importlib.metadata import version
from langchain_core import __version__ as core_version
lg_version = version("langgraph")
# from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import  ChatAnthropic

load_dotenv()

print(f"langchain version: {core_version}")
print(f"langgraph version: {lg_version}")






def main():

    # Test openai
    llm=ChatOpenAI(model='gpt-4o-mini', temperature=0)
    response=llm.invoke("say 'setup completed' in one word")
    print("response from openAI", response)

    # Test Anthropic
    llm_anthropic= ChatAnthropic(model='claude-sonnet-4-5-20250929', temperature=0)
    response_anthropic=llm_anthropic.invoke("say 'Anthropic setup completed' in one word")
    print("response from Anthropic",  response_anthropic)

    print("setup complete")
   


if __name__ == "__main__":
    main()
