
import asyncio

from src.rag_chatbot.rag.blob_utils import chunk_from_blob
from src.rag_chatbot.rag.embedding_utils import process_and_store_chunks
from src.rag_chatbot.rag.RAG_bot import chat_loop
from src.rag_chatbot.rag.env import transcript_container_client, notes_container_client

async def main():
    print("Welcome to the RAG Chat application\n" \
    "Commands:\n" \
    "1. Chunk; generate embedding for chunk and store chunk with embedding in vectorDB for each transcript blob\n" \
    "2. query <your query> - Query the system\n" \
    "3. exit - Exit the application\n")
    while True:
        cmd = int(input("Enter 1, 2 or 3: "))
        if cmd == 1:
            choice = int(input("1: chunk transcripts only\n2: chunk meeting notes only\n3. Chunk both/all blobs\n"))
            transcript_chunks = []
            meeting_note_chunks = []
            if choice == 1:
                transcript_chunks = chunk_from_blob(
                    transcript_container_client,
                    doc_type="transcript",
                    chunk_size=756
                )
            elif choice == 2:
                meeting_note_chunks = chunk_from_blob(
                    notes_container_client,
                    doc_type="meeting_note",
                    chunk_size=400, # smaller may be better for structured note
                    epic_chunking=True,
                    overlap=True # test context chunking for meeting notes
                )
            else:
                print("Invalid choice!")
                continue
            
            all_chunks = transcript_chunks + meeting_note_chunks
            process_and_store_chunks(all_chunks)

        elif cmd == 2:
            user_query = input("Enter your user query").strip()
            response = await chat_loop(user_query)
            print(f"mode: {response["mode"]}")

            if response.get("retrieved", None):
                context_results = response["retrieved"]
                for result in context_results:
                    print(result["id"])
                    # print(result["content"])
                    # print("\n")

            if response.get("grounded_task", None):
                print(response["grounded_task"])
          
            print(response["answer"])
            
                
                
        elif cmd == 3:
            return
        else:
            print("Invalid command try again") 

if __name__ == '__main__':
    asyncio.run(main())
    #main()
    question = """
    Can you create 4 Jira issues based on Reuben's meeting notes where each Epic corresponds to a new Jira issue
    and the title of the epic is the issue summary and the description in the epic is the description of the issue. 
    These issues should be created for the project key of KAN"""