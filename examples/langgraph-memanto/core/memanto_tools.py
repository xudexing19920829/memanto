from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from pydantic import Field

from memanto.cli.client.sdk_client import SdkClient

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
    def _do_setup():
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
        memory_type: MemoryType,
        title: str,
        content: str,
        confidence: float = Field(0.85, ge=0.0, le=1.0),
        tags: str = Field("", description="Comma-separated tags"),
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
        query: str,
        limit: int = Field(5, ge=1, le=20),
        memory_types: str = Field(
            "", description="Optional comma-separated types filter"
        ),
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
    def memanto_answer(question: str) -> str:
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
