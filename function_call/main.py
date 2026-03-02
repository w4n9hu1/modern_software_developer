from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import logging
from tools import get_weather

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Tool function mapping
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
}

def process_tool_call(tool_name: str, tool_args: dict) -> str:
    """Execute the tool and return the result."""
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        result = TOOL_FUNCTIONS[tool_name](**tool_args)
        logger.info(f"  Tool '{tool_name}' executed: {result}")
        return result
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

def main():
    logger.info("Starting function-call demo...\n")

    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather like in New York?"}]

        logger.info("📤 Sending request to model...\n")
        completion = client.chat.completions.create(
            model="kimi-k2-turbo-preview",
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        message = completion.choices[0].message
        tool_calls = message.tool_calls or []

        if tool_calls:
            logger.info(f"🔧 Model requested {len(tool_calls)} tool call(s):\n")
            messages.append(message)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                logger.info(f"  - {tool_name}({json.dumps(tool_args)})")

                tool_result = process_tool_call(tool_name, tool_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

            logger.info("\n📤 Sending tool results back to model...\n")
            follow_up = client.chat.completions.create(
                model="kimi-k2-turbo-preview", messages=messages
            )
            final_response = follow_up.choices[0].message.content
            logger.info(f"✅ Final response:\n{final_response}")
        else:
            logger.info(f"✅ Response (no tools used):\n{message.content}")

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
