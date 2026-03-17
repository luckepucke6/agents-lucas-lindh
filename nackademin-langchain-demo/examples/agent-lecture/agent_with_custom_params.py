from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input


def run():
    model = get_model(temperature=0.3, top_p=0.9)

    agent = create_agent(
        model=model,
        system_prompt=(
            "Du är en hjälpsam assistent som svarar på användarens frågor. "
            "Svara alltid på svenska och var koncis men informativ."
        ),
    )

    user_input = get_user_input("Ställ din fråga")

    process_stream = agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=STREAM_MODES,
    )

    handle_stream(process_stream)


if __name__ == "__main__":
    run()
