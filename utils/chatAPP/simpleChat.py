from typing import List, Dict, Any

from llama_index.core.llms import CustomLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore


def synthesize_messages(messages: List[Dict]):
    """
    Synthesize messages into one history
    Args:
        messages:

    Returns:

    """
    num = 0
    synthesized_messages = ""
    for message in messages:
        synthesized_messages.join(f"Round: {num} - {message['rol']}: {message['content']};")

    return synthesized_messages


def simple_chat(messages: List[Dict], llm: CustomLLM) -> str:
    """
    Simple chat function. This mode has to utilize the message list to form the chat memory.
    Args:
        messages:
        llm:

    Returns:

    """
    """Deal with historical messages"""
    num_messages = len(messages)
    synthesized_messages = ""
    num = 0
    for message in messages[:num_messages - 1]:
        if num % 2 == 0:
            synthesized_messages.join(f"Round: {num / 2}: {message['role']} - {message['content']};")
        else:
            synthesized_messages.join(f"{message['role']} - {message['content']};")
        num += 1
    prompt = (f"You are a helpful assistant. Please based on the chat history and your knowledge to answer the "
              f"question. History: {synthesized_messages}. Now the question is {messages[len(messages) - 1]}, please "
              f"think carefully and response cautiously:")
    response = llm.complete(prompt)
    return response.text


def rag_chat(prompt: str, llm: CustomLLM, memory_token_limit=3900) -> Any:
    """Step 1: create a store to store chat history"""
    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=memory_token_limit,
        chat_store=chat_store,
        chat_store_key="test"
    )
    """Step 2: initialize chat engine or  to run the prompt"""
