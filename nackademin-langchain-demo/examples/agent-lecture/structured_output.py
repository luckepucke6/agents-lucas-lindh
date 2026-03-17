import json

from util.models import get_model
from util.pretty_print import get_user_input, Colors

MOVIE_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Filmens titel"},
        "genre": {"type": "string", "description": "Filmens genre, t.ex. action, drama, komedi"},
        "rating": {"type": "integer", "description": "Betyg från 1 till 10", "minimum": 1, "maximum": 10},
        "summary": {"type": "string", "description": "En kort sammanfattning av filmen i 1-2 meningar"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista med filmens styrkor",
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista med filmens svagheter",
        },
    },
    "required": ["title", "genre", "rating", "summary", "strengths", "weaknesses"],
}


def run():
    model = get_model()

    structured_model = model.with_structured_output(MOVIE_REVIEW_SCHEMA)

    user_input = get_user_input("Vilken film vill du ha en recension av?")

    result = structured_model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Du är en filmkritiker. Analysera filmen som användaren anger "
                    "och ge en strukturerad recension. Svara alltid på svenska."
                ),
            },
            {"role": "user", "content": user_input},
        ]
    )

    raw_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(f"{Colors.DIM}{raw_json}{Colors.RESET}")


if __name__ == "__main__":
    run()
