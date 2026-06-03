from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import Field

from memanto.cli.client.sdk_client import SdkClient

VALID_MEMORY_TYPES = (
    "fact (objective truths/data), "
    "preference (user likes/dislikes), "
    "goal (objectives/targets), "
    "decision (choices made/agreed upon), "
    "artifact (files/code/deliverables), "
    "learning (insights/lessons learned), "
    "event (occurrences/meetings), "
    "instruction (how-tos/directives), "
    "relationship (connections between entities), "
    "context (background info/state), "
    "observation (trends/patterns/notices), "
    "commitment (promises/next steps), "
    "error (failures/mistakes)"
)

MemoryType = Literal[
    "fact",
    "preference",
    "goal",
    "decision",
    "artifact",
    "learning",
    "event",
    "instruction",
    "relationship",
    "context",
    "observation",
    "commitment",
    "error",
]


def create_memanto_tools(client: SdkClient, agent_id: str):
    import threading
    import time
    
    _setup_lock = threading.Lock()

    def _do_setup():
        with _setup_lock:
            try:
                client.create_agent(agent_id=agent_id, pattern="tool")
            except Exception:
                pass
            try:
                client.activate_agent(agent_id, duration_hours=6)
            except Exception:
                pass

    @tool
    def memanto_remember(
        memory_type: Annotated[MemoryType, Field(
            description=(
                f"The semantic type of memory to store. Must be exactly one of the types "
                f"(without the description): fact, preference, goal, decision, artifact, "
                f"learning, event, instruction, relationship, context, observation, "
                f"commitment, or error. Context definitions: {VALID_MEMORY_TYPES}"
            )
        )],
        title: Annotated[str, Field(description="A short, descriptive title for the memory (max 100 chars).")],
        content: Annotated[str, Field(description="The actual information or fact to remember.")],
        confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence level in this memory (0.0 to 1.0).")] = 0.85,
        tags: Annotated[str, Field(description="Optional comma-separated tags for filtering (e.g., 'user_pref, setup').")] = "",
    ) -> str:
        """
        Store a structured memory in Memanto for long-term persistence.
        Use this for information that should survive across different threads and sessions.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        try:
            result = client.remember(
                agent_id=agent_id,
                memory_type=memory_type,
                title=title,
                content=content,
                confidence=confidence,
                tags=tag_list,
                source="langgraph-agent",
            )
        except Exception:
            _do_setup()
            result = client.remember(
                agent_id=agent_id,
                memory_type=memory_type,
                title=title,
                content=content,
                confidence=confidence,
                tags=tag_list,
                source="langgraph-agent",
            )

        return f"Memory stored: {result['memory_id']}"

    @tool
    def memanto_recall(
        query: Annotated[str, Field(description="The natural language search query.")],
        limit: Annotated[int, Field(ge=1, le=20, description="Max number of memories to retrieve.")] = 5,
        memory_types: Annotated[str, Field(description="Optional comma-separated types filter (e.g. 'fact,preference').")] = "",
    ) -> str:
        """
        Search Memanto's persistent memory using natural language.
        Returns stored memories from previous interactions or different agents.
        """
        type_list = (
            [t.strip() for t in memory_types.split(",") if t.strip()]
            if memory_types
            else None
        )

        try:
            result = client.recall(
                agent_id=agent_id,
                query=query,
                limit=limit,
                type=type_list,
            )
        except Exception:
            _do_setup()
            result = client.recall(
                agent_id=agent_id,
                query=query,
                limit=limit,
                type=type_list,
            )

        memories = result.get("memories", [])
        if not memories:
            return f"No memories found for: '{query}'"

        output = [f"Found {len(memories)} memories:"]
        for i, mem in enumerate(memories, 1):
            output.append(
                f"{i}. [{mem.get('type')}] {mem.get('title')}: {mem.get('content')}"
            )

        return "\n".join(output)

    @tool
    def memanto_answer(question: Annotated[str, Field(description="The question to ask the memory system.")]) -> str:
        """
        Ask a question and get an AI-generated answer grounded in the agent's long-term memory.
        Uses RAG to synthesize insights from multiple stored memories.
        """
        try:
            result = client.answer(agent_id=agent_id, question=question)
        except Exception:
            _do_setup()
            result = client.answer(agent_id=agent_id, question=question)

        return f"Answer: {result.get('answer')}"

    return [memanto_remember, memanto_recall, memanto_answer]
