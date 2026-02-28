
from src.rag_chatbot.rag.blob_utils import chunk_transcripts_from_blob
from src.rag_chatbot.rag.embedding_utils import process_and_store_chunks
from src.rag_chatbot.rag.RAG_bot import generate_response
from src.rag_chatbot.rag.retrieval_utils import retrieve_context
from src.rag_chatbot.rag.env import transcript_container_client, notes_container_client

def main():
    print("Welcome to the RAG Chat application\n" \
    "Commands:\n" \
    "1. Chunk; generate embedding for chunk and store chunk with embedding in vectorDB for each transcript blob\n" \
    "2. query <your query> - Query the system\n" \
    "3. exit - Exit the application\n")
    while True:
        cmd = int(input("Enter 1, 2 or 3: "))
        if cmd == 1:
            # choice = int(input("1: chunk transcripts\n2: chunk meeting nptes"))
            # if choice == 1:
                
            # elif choice == 2:
                
            # else:
            #     print("Invalid choice!")
            transcript_chunks = chunk_transcripts_from_blob(
                    transcript_container_client,
                    doc_type="transcript",
                    chunk_size=756
                )
            
            meeting_note_chunks = chunk_transcripts_from_blob(
                    notes_container_client,
                    doc_type="meeting_note",
                    chunk_size=500  # smaller may be better for structured notes
                )
            all_chunks = transcript_chunks + meeting_note_chunks
            process_and_store_chunks(all_chunks)
        elif cmd == 2:
            user_query = input("Enter your user query").strip()
            context_results = retrieve_context(user_query)
            for result in context_results:
                print(result["id"])
                
            answer = generate_response(context_results, user_query)
            print(answer)
        elif cmd == 3:
            return
        else:
            print("Invalid command try again") 

if __name__ == '__main__':
    main()