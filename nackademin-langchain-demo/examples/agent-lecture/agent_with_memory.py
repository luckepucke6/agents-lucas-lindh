from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input


def run():
    # Get predefined attributes
    model = get_model()

    # Create memory saver
    checkpointer = InMemorySaver()
    thread_id = "conversation_001"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Create agent
    agent = create_agent(
        model=model,
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor. "
            "Svara alltid på svenska och var koncis men informativ. "
        ),
        checkpointer=checkpointer,
    )

    # Conversation loop
    while True:
        # Get user input
        user_input = get_user_input("Ställ din fråga")

        # Call the agent
        process_stream = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode=STREAM_MODES,
        )

        # Stream the process
        handle_stream(process_stream)


if __name__ == "__main__":
    run()
