from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.rag_chatbot.rag.RAG_bot import chat_loop
from uuid import uuid4
from redis.exceptions import ConnectionError as RedisConnectionError
from src.rag_chatbot.redis.redis_chat_store import (
    create_chat,
    chat_exists,
    get_messages,
    append_message,
    list_chats,
    update_chat_title,
)
from src.rag_chatbot.redis.redis_client import get_redis

router = APIRouter()

class ChatCreateOut(BaseModel):
    id: str
    title: str

class ChatIn(BaseModel):
    message: str

@router.post("/chats", response_model=ChatCreateOut)
async def create_new_chat(rdb = Depends(get_redis)):
    chat_id = str(uuid4())[:8]
    await create_chat(rdb, chat_id, title="New Chat")
    return {"id": chat_id, "title": "New Chat"}

@router.get("/chats")
async def get_chats(rdb = Depends(get_redis)):
    try:
        return await list_chats(rdb)
    except RedisConnectionError:
        raise HTTPException(status_code=503, detail="Redis is not running")

@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str, rdb = Depends(get_redis)):
    if not await chat_exists(rdb, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return await get_messages(rdb, chat_id)

@router.post("/chats/{chat_id}")
async def chat(chat_id: str, chat_in: ChatIn, rdb = Depends(get_redis)):
    if not await chat_exists(rdb, chat_id):
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} does not exist")

    # load chat history from redis
    history = await get_messages(rdb, chat_id)

    
    await append_message(rdb, chat_id, "user", chat_in.message)

    result = await chat_loop(chat_in.message, history)

    answer = result["answer"] if isinstance(result, dict) else str(result)

    # stores assistant message in redis
    await append_message(rdb, chat_id, "assistant", answer)

    if len(history) == 0:
        title = chat_in.message[:40].strip()
        if title:
            await update_chat_title(rdb, chat_id, title)

    return result