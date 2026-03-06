from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_weather(city: str, unit: str = "celsius") -> str:
    """Fetch weather information for a given city."""
    # Mock weather data for demo purposes.
    sample_data = {
        "beijing": {"temp_c": 22, "condition": "sunny"},
        "shanghai": {"temp_c": 25, "condition": "cloudy"},
        "shenzhen": {"temp_c": 28, "condition": "rainy"},
    }
    key = city.strip().lower()
    data = sample_data.get(key, {"temp_c": 20, "condition": "clear"})
    temp = data["temp_c"] if unit == "celsius" else int(data["temp_c"] * 9 / 5 + 32)
    unit_text = "C" if unit == "celsius" else "F"
    return json.dumps(
        {
            "city": city,
            "temperature": f"{temp}{unit_text}",
            "condition": data["condition"],
        },
        ensure_ascii=False,
    )


client = OpenAI(
    api_key=require_env("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

model_name = require_env("MODEL_NAME")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions for a specified city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit (celsius or fahrenheit)",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "Please check the weather in Beijing and provide a travel recommendation.",
    }
]

first_response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print("=== First Response ===")
print(f"Content: {first_response.choices[0].message.content}")
print(f"Tool Calls: {first_response.choices[0].message.tool_calls}")
print()

assistant_message = first_response.choices[0].message
tool_calls = assistant_message.tool_calls or []

if not tool_calls:
    print(assistant_message.content)
    raise SystemExit(0)

messages.append(
    {
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }
)

for tool_call in tool_calls:
    if tool_call.function.name != "get_weather":
        continue

    args = json.loads(tool_call.function.arguments or "{}")
    result = get_weather(
        city=args.get("city", ""),
        unit=args.get("unit", "celsius"),
    )
    print("=== Tool Call Result ===")
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {args}")
    print(f"Result: {result}")
    print()
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        }
    )

final_response = client.chat.completions.create(
    model=model_name,
    messages=messages,
)

print("=== Final Response ===")
print(final_response.choices[0].message.content)
