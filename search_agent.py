
from typing import TypedDict, Annotated
import operator

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage

from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph


tool = TavilySearchResults(max_results=4) #increased number of results
print(type(tool))
print(tool.name)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system_message=""):
        self.system_message = system_message
        graph=StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph =  graph.compile()

        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(list(self.tools.values()))


    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0


    def take_action(self, state: AgentState):
        tool_calls = state['messages'] [-1] .tool_calls
        results = []
        for t in tool_calls:
            print(f" {'*' * 10 }--calling {t}")
            if not t['name'] in self.tools:
                print(f" no tool called {t}")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            # print(f"Tool response {result}")
        print(f"{'*' * 10 }--Back to the model!")
        return {'messages': results}

    def call_model(self,state: AgentState):
        messages=state["messages"]
        if self.system_message:
            messages=[SystemMessage(content=self.system_message)]+messages
        print(f"{'*' * 10 }--calling model {messages[-1].content}")
        message=self.model.invoke(messages)
        print(f"{'*' * 10 }-- LLM response {message}")
        return {'messages': [message]}

model = ChatOpenAI(
            model="gpt-oss:120b-cloud",  # Or any model you pulled (e.g., mistral)
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Required but ignored
            temperature=0
        )
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
from IPython.display import Image
abot = Agent(model, [tool], system_message=prompt)
messages = [HumanMessage(content="Population of Peddakorpole")]
result = abot.graph.invoke({"messages": messages})

print("Results "+ ('*' * 10 ))

print(result['messages'][-1].content)
