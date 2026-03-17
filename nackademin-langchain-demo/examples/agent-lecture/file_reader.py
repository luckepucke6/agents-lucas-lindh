from langchain.agents import create_agent
from langchain_core.tools import tool

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input

@tool
def read_file(file_path: str) -> str:
    """Läser innehållet från ett dokument på datorn.
    
    Args:
        file_path: Sökvägen till filen som ska läsas (t.ex. '/path/to/document.txt')
    
    Returns:
        Innehållet i filen som en sträng
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Fel: Filen '{file_path}' hittades inte."
    except PermissionError:
        return f"Fel: Ingen behörighet att läsa filen '{file_path}'."
    except Exception as e:
        return f"Fel vid läsning av fil: {str(e)}"

def run():
    # Get predefined attributes
    model = get_model()
    # Create agent
    agent = create_agent(
        model=model,
        tools=[read_file],
        system_prompt="Du är en hjälpsam assistent som kan arbeta med filer och mappar på datorn. ",
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
