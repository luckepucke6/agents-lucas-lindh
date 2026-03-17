from langchain.agents import create_agent
from langchain_core.tools import tool

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

@tool
def counting_characters(text: str) -> int:
    """Räknar antalet tecken i en text."""
    return len(text)


def run():
    # Get predefined attributes
    model = get_model()

    # Define tools
    tools = [counting_characters]

    # Create agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor."
        ),
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
