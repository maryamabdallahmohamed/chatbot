from dotenv import load_dotenv
import os
import google.generativeai as genai
from tavily import TavilyClient
import re

# Load environment variables
load_dotenv()
api = os.getenv("gemini_key")
tabily_key=os.getenv("TAVILY")

# Configure Gemini client
genai.configure(api_key=api)



system_prompt = """
You are an AI assistant specialized in enhancing prompts and optionally using external tools.

## Role
1. Take user-input prompts and transform them into more engaging, detailed, and thought-provoking questions.
2. Clearly describe the process you follow to enhance a prompt and the types of improvements you make.
3. Provide an enriched, multi-layered version of the prompt that encourages deeper thinking and more insightful responses.

## Tool Usage
- If you need more information to enrich the prompt, you may PAUSE and call a tool.
- When calling a tool, respond **only** in the following format:

PAUSE
Action: <tool_name>: <tool_input>

Example:
PAUSE
Action: search: "latest research on AI in education"

- After receiving an Observation from a tool, continue reasoning with the new information.
- When you are ready to provide the final enriched prompt, output it using:

Answer: <your enriched prompt here>

## Rules
- Always follow the PAUSE/Action/Observation/Answer flow when tools are needed.
- If no tool is needed, go directly to `Answer`.
- Never invent tool names outside those provided.
- Available tool(s): search
""".strip()


def search_online(query):
    tavily_client = TavilyClient(api_key=tabily_key)
    response = tavily_client.search(
        query=query,
        days=7,
        max_results=3,
        include_answer=True,
        include_domains=['.org', '.edu', '.eg', '.gov']
    )
    return response

class Agent:
    def __init__(self, system=None):
        self.system = system
        self.messages = []
        
        if self.system is not None:
            self.messages.append({"role": "system", "content": self.system})

        # Create Gemini model client once
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def __call__(self, message):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        # Concatenate history as a single string (Gemini expects text input, not ChatML dicts)
        conversation = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.messages])
        
        response = self.model.generate_content(conversation)
        return response.text


def agent_loop(max_iterations, query, prompt):
    tools = {
        "search": search_online,
    }
    agent = Agent(system=system_prompt)
    next_prompt = prompt

    for i in range(max_iterations):
        result = agent(next_prompt)
        print(f"\n🤖 Agent step {i+1}:\n{result}\n")

        # Check if model is asking to use a tool
        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            if action:
                chosen_tool, arg = action[0]
                if chosen_tool in tools:
                    result_tool = tools[chosen_tool](arg)
                    next_prompt = f"Observation: {result_tool}"
                else:
                    next_prompt = "Observation: Tool not found"
                continue

        # If model signals final answer
        if "Answer" in result:
            print("\n✅ Final Answer Found:\n", result)
            print(agent.messages)
            break



if __name__=="__main__":
    agent_loop(
        max_iterations=5,
        query="What is AI?",
        prompt="Explain what AI is and its impact on education. If you need more info, use search."
    )
