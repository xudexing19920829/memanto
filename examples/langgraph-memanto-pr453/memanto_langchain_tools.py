from typing import List
from langchain_core.tools import tool
from memanto.cli.client.sdk_client import SdkClient

def create_memanto_tools(client: SdkClient, agent_id: str) -> List[callable]:
    """
    Creates LangChain tools bound to a specific Memanto client and agent_id.
    """
    
    @tool
    def memanto_remember(content: str, memory_type: str = "fact") -> str:
        """
        Store a new memory into the agent's long-term semantic database.
        Use this tool to save facts, preferences, or important instructions that you must remember for future sessions.
        
        Args:
            content: The text of the memory to store (e.g. "User prefers dark mode").
            memory_type: The category of the memory. Valid types: 'fact', 'preference', 'decision', 'instruction', 'goal'.
        """
        try:
            # We must use a short title for the memory
            title = content[:47] + "..." if len(content) > 50 else content
            result = client.remember(
                agent_id=agent_id,
                memory_type=memory_type,
                title=title,
                content=content
            )
            return f"Successfully remembered: '{content}' (Type: {memory_type})"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    @tool
    def memanto_recall(query: str) -> str:
        """
        Run a semantic search against the agent's long-term memories.
        Use this tool whenever you need context about the user's past preferences or facts from previous conversations.
        
        Args:
            query: The search query to retrieve relevant memories.
        """
        try:
            result = client.recall(agent_id=agent_id, query=query)
            memories = result.get("memories", [])
            
            if not memories:
                return "No relevant memories found."
            
            formatted = []
            for mem in memories:
                # mem is likely a dict
                content = mem.get("content", str(mem))
                formatted.append(f"- {content}")
                
            return "Retrieved Memories:\n" + "\n".join(formatted)
        except Exception as e:
            return f"Error recalling memory: {str(e)}"
            
    return [memanto_remember, memanto_recall]
