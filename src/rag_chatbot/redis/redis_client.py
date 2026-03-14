import redis.asyncio as redis

def get_redis():
    """
    Creates a Redis client conntection
    """
    return redis.Redis(
        host="redis", # changed to service name in docker-compose
        port=6379,
        decode_responses=True, # redis returns str instead of bytes
    )