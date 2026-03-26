
import asyncio
from src.rag_chatbot.rag.blob_utils import chunk_from_blob
from src.rag_chatbot.rag.embedding_utils import process_and_store_chunks
from src.rag_chatbot.rag.env import transcript_container_client, notes_container_client

async def embed_chunks():
    """
    method designed to be run from the command line. First chunks all the blobs from a specific container, user can chose to chunk
    all the blob's from the transcript container or meeting container or both.

    Then chunks are stored in their corresponding azure AI search index. So transcript_container blobs are stored in the transcripts
    container for example.
    """
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

    all_chunks = transcript_chunks + meeting_note_chunks
    process_and_store_chunks(all_chunks)    

if __name__ == '__main__':
    asyncio.run(embed_chunks())
