from azure.storage.blob import BlobServiceClient, ContainerClient
import os
import pandas as pd 
from dotenv import load_dotenv

load_dotenv('.env')

BLOB_SAS_URL = os.getenv("BLOB_SAS_URL")
container_name = "transcripts"
folder_path = "earnings_calls"  # virtual folder 
EARNING_CALL_URL = "hf://datasets/yeong-hwan/2024-earnings-call-transcript/2024-earnings-call-transcripts.jsonl"

container_client = ContainerClient.from_container_url(BLOB_SAS_URL)


def save_formatted_transcript_local(transcript: str):
    try:
        with open("data/2024-earning-call-transcript.txt", "wt", encoding="utf-8") as f:
            f.write(transcript)
    except FileNotFoundError as e:
        print(e)

def format_transcript(transcript: list) -> str:
    """Method should return transcript in a format similar to that of 
    a .vtt file transcript from Teams to allow swapping of data in the future
    format:
    00:00:23.123 --> 00:00:28.843 (time speaker speaks for)
    <v Alex Hanna (speaker name))>This is a test for meeting transcriptions.
    This is about to be over.</v>"""
    """
    transcript: list
        list of what each speaker says followed one after the other
    """
    speakers = set()
    transcript_txt = ""
    lines = []
    for speech in transcript:
        print(speech)
        speaker = speech.get('speaker')
        content = speech.get('content')
        speakers.add(speaker)
        lines.append(f"<v {speaker}>{content}</v>")
    transcript_txt = "\n".join(lines)
    return transcript_txt



def add_meta_data(metadata: dict, blob_name: str):
    blob_client = container_client.get_blob_client(blob_name)

    metadata = {
        "meetingDate": "2025-11-30",
        "department": "Engineering",
        "transcriptType": "SprintReview"
    }

    blob_client.set_blob_metadata(metadata)


def read_in_transcript(url: str = EARNING_CALL_URL) -> pd.DataFrame:
    df = pd.read_json(url, lines=True)
    #df.to_csv("data/2024-earnings-call-transcript.csv", index=False)
    df.to_json("data/2024-earnings-call-transcript.json",
                orient="records", lines=True)
    return df


def send_to_storage(ind = 5):
    blob_name = f"{folder_path}/"
    

    #df = pd.read_csv("data/2024-earnings-call-transcript.csv")
    df = pd.read_json("data/2024-earnings-call-transcript.json", lines=True)
    # get a specific call's transcript-
    transcript = df.loc[ind, "conversations"]
    formatted_transcript = format_transcript(transcript)

    # name format: ticker-date-quarter.txt
    # e.g: aapl-2024-q4.txt
    ticker = str(df.loc[ind, "ticker"]).strip()
    date = str(df.loc[ind, "year"]).strip()
    quarter = str(df.loc[ind, "q"]).strip()
    transcript_name = f"{ticker}-{date}-{quarter}.txt"


    blob_name += transcript_name
    container_client.upload_blob(
            name=blob_name,
            data=formatted_transcript,
            overwrite=True  # set to False to avoid overwriting
    )

    print(f"Uploaded '{transcript_name}' to container '{container_name}' at path '{blob_name}'")

