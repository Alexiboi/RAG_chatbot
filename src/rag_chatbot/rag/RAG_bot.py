
from src.rag_chatbot.rag.retrieval_utils import retrieve_context
from src.rag_chatbot.rag.env import deployment_name, client


def generate_response(context: list[dict], user_query: str) -> str:
    context_texts = [doc["content"] for doc in context]

    context_block = "\n\n---\n\n".join(context_texts)
    final_prompt = f"""
    You are an assistant that answers questions using the transcript context below.
    If the answer is not in the context, say that the transcript does not contain the information.

    Context:
    {context_block}

    User question:
    {user_query}

    Answer:
    """
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": final_prompt}
        ],
        #temperature=0.2
    )

    return response.choices[0].message.content



def generate_contextualized_response(inputs: dict) -> dict:
    user_query = inputs["question"]

    user_query = user_query.strip()
    context_results = retrieve_context(user_query)
    answer = generate_response(context_results, user_query)
    return {
        "answer": answer,
        "prompt": user_query,
        "retrieved": context_results
    }
