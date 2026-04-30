from azure.storage.blob import ContainerClient
import pandas as pd 
from src.backend.rag.env import TRANSCRIPT_SAS_URL

folder_path = "earnings_calls"  # virtual folder 


def format_transcript(transcript: list) -> str:
    """Method should return transcript in a format similar to that of 
    a .vtt file transcript from Teams to allow swapping of data in the future
    format:
    00:00:23.123 --> 00:00:28.843 (time speaker speaks for)
    <v Alex Hanna (speaker name))>This is a test for meeting transcriptions.
    This is about to be over.</v>"""
    """
    Args:  
        transcript (list): list of what each speaker says followed one after the other
    Returns:
        str: The entire transcript formatted in the desired way (described before) 

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



def read_in_transcript(url: str) -> pd.DataFrame:
    """
    One off method used to read in public dataset of earnings calls in json format and upload them to a data folder locally.
    Public earning calls dataset available at: hf://datasets/yeong-hwan/2024-earnings-call-transcript/2024-earnings-call-transcripts.jsonl 

    Reads in json and converts to pandas df.

    Args:
        url (str): url of json dataset to read in
    """
    df = pd.read_json(url, lines=True)
    #df.to_csv("data/2024-earnings-call-transcript.csv", index=False)
    df.to_json("data/2024-earnings-call-transcript.json",
                orient="records", lines=True)
    return df


def send_to_storage(container_client_url: str, data_file: str, ind: int = 5):
    """
    Converts a json earning call transcript into a formatted string similar to a .vtt file of a transcript from Microsoft Teams
    and uploads it to a specified container client in Azure blob storage

    Args:
        container_client_url (str): The ""SAS"" url of the container client you wish to upload to in Azure Blob storage
        data_file (str): local json data file (file path)
        ind (int): starting index of first record from data_file df you want to upload 
    """
    blob_name = f"{folder_path}/"
    
    container_client = ContainerClient.from_container_url(container_client_url)

    df = pd.read_json(data_file, lines=True)
   
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

    print(f"Uploaded '{transcript_name}' to container '{container_client.container_name}' at path '{blob_name}'")

if __name__ == "__main__":
    #pass
    send_to_storage(TRANSCRIPT_SAS_URL, "./data/2024-earnings-call-transcript.json")

