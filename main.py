from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
## defining the calculator tool
def calculator(a: float, b: float) -> str:
     """Performs basic arithmetic operations between two numbers a and b."""
     return f"The sum of {a} and {b} is {a + b}. The difference when subtracting {b} from {a} is {a - b}. The product of {a} and {b} is {a * b}. The quotient when dividing {a} by {b} is {a / b}."

def main():
    ## initializing the language model
    model = ChatOpenAI(temperature=0)
    
    tools = [calculator]
    ## creating the React agent with the calculator tool
    agent_executor = create_react_agent(model, tools)

    print("Welcome to your AI companion! Type 'exit' to quit.")
    print("You can ask me to perform calculations for you or just ask a question.")

    ## main interaction loop
    while True:
        ## getting user input
        user_input = input("\nYou: ").strip()
        if user_input == 'exit':
            print("Goodbye!")
            break

        print("\nAssistant: ", end ="")
        ## streaming the agent's response
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
            ):
                ## printing the response chunk by chunk
                if "agent" in chunk and "messages" in chunk["agent"]:
                    ## printing each message content
                    for message in chunk["agent"]["messages"]:
                        print(message.content, end="")
                print()
        
if __name__ == "__main__":
    main()