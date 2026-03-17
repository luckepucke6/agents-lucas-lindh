from langchain.agents import create_agent
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

def run():
    # Get predefined attributes
    model = get_model()
    
    # Create Requests Toolkit for fetching web pages (for requests_get tool)
    tools = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=True,  # Required for using the tools
    ).get_tools()
        
    # Create agent with tools
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "Du är en hjälpsam assistent som kan hämta innehåll från webbsidor. "
            "Du kan hämta data från webbsidor. "
        )
    )   
    # Get user input
    user_input = get_user_input("Ställ din fråga")
        
    # Call the agent
    process_stream = agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=STREAM_MODES,
    )
    # Stream the process
    handle_stream(process_stream)

if __name__ == "__main__":
    run()
