import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.rag_chatbot.rag.embedding_utils import (
    generate_embeddings,
    process_and_store_chunks,
    get_search_client_for_doc_type,
    make_chunk_id,
    extract_metadata,
    extract_earning_call_metadata,
    extract_meeting_note_metadata,
    COMPANY_MAP,
)


# ==================== FIXTURES ====================

@pytest.fixture
def sample_transcript_chunk():
    """Sample transcript chunk for testing."""
    return {
        "source": "a-2024-1.txt",
        "chunk_id": "chunk-1",
        "content": "Q1 2024 earnings call transcript for Agilent",
        "docType": "transcript",
    }


@pytest.fixture
def sample_meeting_note_chunk():
    """Sample meeting note chunk for testing."""
    return {
        "source": "meeting-notes/2026-01-28-john-notes.txt",
        "chunk_id": "chunk-2",
        "content": "Meeting notes from January 28, 2026",
        "docType": "meeting_note",
    }


@pytest.fixture
def sample_chunks_mixed(sample_transcript_chunk, sample_meeting_note_chunk):
    """Multiple chunks of different types."""
    return [sample_transcript_chunk, sample_meeting_note_chunk]


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors."""
    return [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


@pytest.fixture
def mock_success_result():
    """Mock successful indexing result."""
    result = Mock()
    result.succeeded = True
    result.key = "doc-1"
    return result


@pytest.fixture
def mock_failed_result():
    """Mock failed indexing result."""
    result = Mock()
    result.succeeded = False
    result.key = "doc-2"
    result.error_message = "Upload failed: duplicate key"
    return result


@pytest.fixture
def mock_search_client():
    """Mock search client."""
    client = Mock()
    client._index_name = "test-index"
    return client


# ==================== TEST GENERATE_EMBEDDINGS ====================

class TestGenerateEmbeddings:
    """Unit tests for generate_embeddings function."""

    @patch("src.rag_chatbot.rag.embedding_utils.EMBEDDING_CLIENT")
    def test_returns_list_of_embeddings(self, mock_embedding_client):
        """Test that function returns list of embeddings."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_embedding_client.embeddings.create.return_value = mock_response

        texts = ["hello world", "test text"]
        result = generate_embeddings(texts)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("src.rag_chatbot.rag.embedding_utils.EMBEDDING_CLIENT")
    def test_passes_texts_to_embedding_client(self, mock_embedding_client):
        """Test that texts are passed to embedding client."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1])]
        mock_embedding_client.embeddings.create.return_value = mock_response

        texts = ["hello", "world"]
        generate_embeddings(texts)

        call_args = mock_embedding_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == texts

    @patch("src.rag_chatbot.rag.embedding_utils.EMBEDDING_CLIENT")
    def test_handles_single_text(self, mock_embedding_client):
        """Test that function handles single text."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_embedding_client.embeddings.create.return_value = mock_response

        result = generate_embeddings(["single text"])

        assert len(result) == 1


# ==================== TEST GET_SEARCH_CLIENT_FOR_DOC_TYPE ====================

class TestGetSearchClientForDocType:
    """Unit tests for get_search_client_for_doc_type function."""

    @patch("src.rag_chatbot.rag.embedding_utils.TRANSCRIPT_SEARCH_CLIENT")
    def test_returns_transcript_client_for_transcript_type(self, mock_transcript_client):
        """Test that transcript type returns TRANSCRIPT_SEARCH_CLIENT."""
        result = get_search_client_for_doc_type("transcript")
        assert result is mock_transcript_client

    @patch("src.rag_chatbot.rag.embedding_utils.MEETING_NOTES_SEARCH_CLIENT")
    def test_returns_meeting_notes_client_for_meeting_note_type(self, mock_meeting_client):
        """Test that meeting_note type returns MEETING_NOTES_SEARCH_CLIENT."""
        result = get_search_client_for_doc_type("meeting_note")
        assert result is mock_meeting_client

    def test_raises_error_for_unknown_doc_type(self):
        """Test that unknown doc_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown doc_type"):
            get_search_client_for_doc_type("unknown_type")


# ==================== TEST MAKE_CHUNK_ID ====================

class TestMakeChunkId:
    """Unit tests for make_chunk_id function."""

    def test_creates_stable_id(self):
        """Test that same inputs produce same ID."""
        id1 = make_chunk_id("test-file.txt", "content", "transcript")
        id2 = make_chunk_id("test-file.txt", "content", "transcript")
        assert id1 == id2

    def test_different_content_produces_different_id(self):
        """Test that different content produces different ID."""
        id1 = make_chunk_id("test-file.txt", "content1", "transcript")
        id2 = make_chunk_id("test-file.txt", "content2", "transcript")
        assert id1 != id2

    def test_different_source_produces_different_id(self):
        """Test that different source produces different ID."""
        id1 = make_chunk_id("file1.txt", "content", "transcript")
        id2 = make_chunk_id("file2.txt", "content", "transcript")
        assert id1 != id2

    def test_different_doc_type_produces_different_id(self):
        """Test that different doc_type produces different ID."""
        id1 = make_chunk_id("file.txt", "content", "transcript")
        id2 = make_chunk_id("file.txt", "content", "meeting_note")
        assert id1 != id2

    def test_id_starts_with_doc_type(self):
        """Test that ID starts with doc_type."""
        doc_id = make_chunk_id("file.txt", "content", "transcript")
        assert doc_id.startswith("transcript-")

    def test_id_contains_sanitized_source_name(self):
        """Test that ID contains sanitized source name."""
        doc_id = make_chunk_id("a-2024-1.txt", "content", "transcript")
        assert "a-2024-1" in doc_id

    def test_sanitizes_special_characters(self):
        """Test that special characters are sanitized."""
        doc_id = make_chunk_id("file@#$%.txt", "content", "transcript")
        assert "@" not in doc_id
        assert "#" not in doc_id
        assert "$" not in doc_id
        assert "%" not in doc_id


# ==================== TEST EXTRACT_EARNING_CALL_METADATA ====================

class TestExtractEarningCallMetadata:
    """Unit tests for extract_earning_call_metadata function."""

    def test_extracts_metadata_from_valid_filename(self):
        """Test extraction from valid filename pattern."""
        metadata = extract_earning_call_metadata("a-2024-1.txt")
        
        assert metadata["docType"] == "earnings_call"
        assert metadata["company"] == "Agilent"
        assert metadata["year"] == 2024
        assert metadata["quarter"] == 1

    def test_all_company_codes(self):
        """Test extraction for all company codes."""
        test_cases = {
            "a": "Agilent",
            "aapl": "Apple",
            "amzn": "Amazon",
            "bx": "BlackStone",
        }
        
        for code, company in test_cases.items():
            metadata = extract_earning_call_metadata(f"{code}-2024-1.txt")
            assert metadata["company"] == company

    def test_extracts_all_quarters(self):
        """Test extraction for all quarters."""
        for quarter in [1, 2, 3, 4]:
            metadata = extract_earning_call_metadata(f"a-2024-{quarter}.txt")
            assert metadata["quarter"] == quarter

    def test_extracts_report_date_in_iso_format(self):
        """Test that report date is in ISO format with Z suffix."""
        metadata = extract_earning_call_metadata("a-2024-1.txt")
        
        assert metadata["reportDate"].endswith("Z")
        # Q1 -> month 3 (first month of quarter)
        assert "2024-03-01" in metadata["reportDate"]

    def test_report_date_for_q2(self):
        """Test report date for Q2."""
        metadata = extract_earning_call_metadata("a-2024-2.txt")
        assert "2024-06-01" in metadata["reportDate"]

    def test_report_date_for_q3(self):
        """Test report date for Q3."""
        metadata = extract_earning_call_metadata("a-2024-3.txt")
        assert "2024-09-01" in metadata["reportDate"]

    def test_report_date_for_q4(self):
        """Test report date for Q4."""
        metadata = extract_earning_call_metadata("a-2024-4.txt")
        assert "2024-12-01" in metadata["reportDate"]

    def test_raises_error_for_invalid_filename_format(self):
        """Test that invalid filename raises ValueError."""
        invalid_filenames = [
            "invalid.txt",
            "a-2024.txt",
            "a-2024-5.txt",  # quarter > 4
            "2024-1-a.txt",  # wrong order
        ]
        
        for filename in invalid_filenames:
            with pytest.raises(ValueError, match="Invalid transcript format"):
                extract_earning_call_metadata(filename)

    def test_raises_error_for_unknown_company_code(self):
        """Test that unknown company code raises ValueError."""
        with pytest.raises(ValueError, match="Unknown company code"):
            extract_earning_call_metadata("xyz-2024-1.txt")

    def test_handles_full_paths(self):
        """Test extraction from full file paths."""
        metadata = extract_earning_call_metadata("data/transcripts/a-2024-1.txt")
        
        assert metadata["company"] == "Agilent"
        assert metadata["year"] == 2024


# ==================== TEST EXTRACT_MEETING_NOTE_METADATA ====================

class TestExtractMeetingNoteMetadata:
    """Unit tests for extract_meeting_note_metadata function."""

    def test_extracts_doctype(self):
        """Test that docType is set to meeting_note."""
        metadata = extract_meeting_note_metadata("2026-01-28-john-notes.txt")
        assert metadata["docType"] == "meeting_note"

    def test_extracts_author(self):
        """Test that author is extracted."""
        metadata = extract_meeting_note_metadata("2026-01-28-john-notes.txt")
        assert metadata["author"] == "john"

    def test_extracts_author_converts_lowercase(self):
        """Test that author is extracted."""
        metadata = extract_meeting_note_metadata("2026-01-28-John-notes.txt")
        assert metadata["author"] == "john"

    def test_extracts_author_different_name(self):
        """Test that author is extracted."""
        metadata = extract_meeting_note_metadata("2026-01-28-mark-notes.txt")
        assert metadata["author"] == "mark"

    def test_extracts_meeting_date_from_filename(self):
        """Test that meeting date is extracted from filename."""
        metadata = extract_meeting_note_metadata("2026-01-28-john-notes.txt")
        assert metadata["meetingDate"] == "2026-01-28"

    def test_handles_different_date_formats(self):
        """Test extraction with various date formats in filename."""
        test_cases = [
            ("2026-01-15-notes.txt", "2026-01-15"),
            ("2025-12-31-meeting.txt", "2025-12-31"),
            ("2024-06-01-notes.txt", "2024-06-01"),
        ]
        
        for filename, expected_date in test_cases:
            metadata = extract_meeting_note_metadata(filename)
            assert metadata["meetingDate"] == expected_date

    def test_meeting_date_is_none_when_not_found(self):
        """Test that meetingDate is None when no date in filename."""
        metadata = extract_meeting_note_metadata("meeting-notes/john-notes.txt")
        assert metadata["meetingDate"] is None
        assert metadata["author"] == "john"

    def test_author_is_none_when_not_found(self):
        """Test that extraction does not fail and meetingdate can still be extracted when name of file
        excludes author"""
        metadata = extract_meeting_note_metadata("meeting-notes/2026-01-28-notes.txt")
        # Should still extract the date
        assert metadata["meetingDate"] == "2026-01-28"
        assert metadata["author"] is None

    def test_author_and_meetingDate_is_none(self):
        """Test extraction from full paths."""
        metadata = extract_meeting_note_metadata("meeting-notes/notes.txt")
        assert metadata["meetingDate"] is None
        assert metadata["author"] is None

    def test_author_and_meetingDate_swapped(self):
        """Test extraction from full paths."""
        metadata = extract_meeting_note_metadata("meeting-notes/john-2026-01-28-notes.txt")
        assert metadata["meetingDate"] == "2026-01-28"
        assert metadata["author"] == "john"

    def test_handles_full_paths(self):
        """Test extraction from full paths."""
        metadata = extract_meeting_note_metadata("meeting-notes/2026-01-28-john-notes.txt")
        assert metadata["meetingDate"] == "2026-01-28"

    
# ==================== TEST EXTRACT_METADATA ====================

class TestExtractMetadata:
    """Unit tests for extract_metadata function."""

    def test_calls_extract_earning_call_metadata_for_transcript_type(self):
        """Test that transcript type calls earning call extractor."""
        metadata = extract_metadata("a-2024-1.txt", "transcript")
        
        assert metadata["docType"] == "earnings_call"
        assert metadata["company"] == "Agilent"


    def test_raises_error_for_unknown_doc_type(self):
        """Test that unknown doc_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown doc_type"):
            extract_metadata("file.txt", "unknown_type")


# ==================== TEST PROCESS_AND_STORE_CHUNKS ====================

class TestProcessAndStoreChunks:
    """Unit tests for process_and_store_chunks function."""

    @patch("src.rag_chatbot.rag.embedding_utils.get_search_client_for_doc_type")
    @patch("src.rag_chatbot.rag.embedding_utils.ensure_index_exists")
    @patch("src.rag_chatbot.rag.embedding_utils.generate_embeddings")
    @patch("src.rag_chatbot.rag.embedding_utils.extract_metadata")
    @patch("src.rag_chatbot.rag.embedding_utils.make_chunk_id")
    def test_groups_chunks_by_doc_type(
        self,
        mock_make_chunk_id,
        mock_extract_metadata,
        mock_generate_embeddings,
        mock_ensure_index,
        mock_get_client,
        sample_chunks_mixed,
    ):
        """Test that chunks are grouped by docType."""
        mock_make_chunk_id.side_effect = lambda s, c, t: f"id-{s}-{t}"
        mock_extract_metadata.return_value = {}
        mock_generate_embeddings.return_value = [[0.1], [0.2]]
        
        mock_client = Mock()
        mock_client._index_name = "test-index"
        mock_client.upload_documents.return_value = [Mock(succeeded=True)]
        mock_get_client.return_value = mock_client

        result = process_and_store_chunks(sample_chunks_mixed)

        # Should have entries for both transcript and meeting_note
        assert "transcript" in result
        assert "meeting_note" in result

    @patch("src.rag_chatbot.rag.embedding_utils.get_search_client_for_doc_type")
    @patch("src.rag_chatbot.rag.embedding_utils.ensure_index_exists")
    @patch("src.rag_chatbot.rag.embedding_utils.generate_embeddings")
    @patch("src.rag_chatbot.rag.embedding_utils.extract_metadata")
    @patch("src.rag_chatbot.rag.embedding_utils.make_chunk_id")
    def test_calls_generate_embeddings_with_content(
        self,
        mock_make_chunk_id,
        mock_extract_metadata,
        mock_generate_embeddings,
        mock_ensure_index,
        mock_get_client,
        sample_transcript_chunk,
    ):
        """Test that generate_embeddings is called with chunk content."""
        mock_make_chunk_id.return_value = "id-1"
        mock_extract_metadata.return_value = {}
        mock_generate_embeddings.return_value = [[0.1, 0.2]]
        
        mock_client = Mock()
        mock_client._index_name = "test-index"
        mock_client.upload_documents.return_value = [Mock(succeeded=True)]
        mock_get_client.return_value = mock_client

        process_and_store_chunks([sample_transcript_chunk])

        mock_generate_embeddings.assert_called_once()
        call_args = mock_generate_embeddings.call_args[0][0]
        assert sample_transcript_chunk["content"] in call_args

    @patch("src.rag_chatbot.rag.embedding_utils.get_search_client_for_doc_type")
    @patch("src.rag_chatbot.rag.embedding_utils.ensure_index_exists")
    @patch("src.rag_chatbot.rag.embedding_utils.generate_embeddings")
    @patch("src.rag_chatbot.rag.embedding_utils.extract_metadata")
    @patch("src.rag_chatbot.rag.embedding_utils.make_chunk_id")
    def test_calls_ensure_index_exists(
        self,
        mock_make_chunk_id,
        mock_extract_metadata,
        mock_generate_embeddings,
        mock_ensure_index,
        mock_get_client,
        sample_transcript_chunk,
    ):
        """Test that ensure_index_exists is called with index name."""
        mock_make_chunk_id.return_value = "id-1"
        mock_extract_metadata.return_value = {}
        mock_generate_embeddings.return_value = [[0.1]]
        
        mock_client = Mock()
        mock_client._index_name = "transcript-chunks"
        mock_client.upload_documents.return_value = [Mock(succeeded=True)]
        mock_get_client.return_value = mock_client

        process_and_store_chunks([sample_transcript_chunk])

        mock_ensure_index.assert_called_once_with("transcript-chunks")

    @patch("src.rag_chatbot.rag.embedding_utils.get_search_client_for_doc_type")
    @patch("src.rag_chatbot.rag.embedding_utils.ensure_index_exists")
    @patch("src.rag_chatbot.rag.embedding_utils.generate_embeddings")
    @patch("src.rag_chatbot.rag.embedding_utils.extract_metadata")
    @patch("src.rag_chatbot.rag.embedding_utils.make_chunk_id")
    def test_uploads_documents_with_correct_structure(
        self,
        mock_make_chunk_id,
        mock_extract_metadata,
        mock_generate_embeddings,
        mock_ensure_index,
        mock_get_client,
        sample_transcript_chunk,
    ):
        """Test that documents are uploaded with correct structure."""
        mock_make_chunk_id.return_value = "stable-id-123"
        mock_extract_metadata.return_value = {"company": "Agilent", "year": 2024}
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        mock_client = Mock()
        mock_client._index_name = "test-index"
        mock_client.upload_documents.return_value = [Mock(succeeded=True)]
        mock_get_client.return_value = mock_client

        process_and_store_chunks([sample_transcript_chunk])

        mock_client.upload_documents.assert_called_once()
        uploaded_docs = mock_client.upload_documents.call_args[1]["documents"]
        
        assert len(uploaded_docs) == 1
        doc = uploaded_docs[0]
        assert doc["id"] == "stable-id-123"
        assert doc["content"] == sample_transcript_chunk["content"]
        assert doc["embedding"] == [0.1, 0.2, 0.3]
        assert doc["docType"] == "transcript"
        assert doc["company"] == "Agilent"
        assert doc["year"] == 2024

    @patch("src.rag_chatbot.rag.embedding_utils.get_search_client_for_doc_type")
    @patch("src.rag_chatbot.rag.embedding_utils.ensure_index_exists")
    @patch("src.rag_chatbot.rag.embedding_utils.generate_embeddings")
    @patch("src.rag_chatbot.rag.embedding_utils.extract_metadata")
    @patch("src.rag_chatbot.rag.embedding_utils.make_chunk_id")
    def test_handles_failed_indexing_result(
        self,
        mock_make_chunk_id,
        mock_extract_metadata,
        mock_generate_embeddings,
        mock_ensure_index,
        mock_get_client,
        sample_chunks_mixed,
        capsys,
    ):
        """Test that failed indexing results are printed."""
        mock_make_chunk_id.side_effect = lambda s, c, t: f"id-{t}"
        mock_extract_metadata.return_value = {}
        mock_generate_embeddings.return_value = [[0.1], [0.2]]
        
        # Create success and failed results
        success_result = Mock()
        success_result.succeeded = True
        success_result.key = "id-transcript"
        
        failed_result = Mock()
        failed_result.succeeded = False
        failed_result.key = "id-meeting_note"
        failed_result.error_message = "Duplicate key error"
        
        mock_client = Mock()
        mock_client._index_name = "test-index"
        mock_client.upload_documents.side_effect = [
            [success_result],  # First call for transcript
            [failed_result],   # Second call for meeting_note
        ]
        mock_get_client.return_value = mock_client

        result = process_and_store_chunks(sample_chunks_mixed)

        # Check that failure was printed
        captured = capsys.readouterr()
        assert "Failed to index id-meeting_note: Duplicate key error" in captured.out

    @patch("src.rag_chatbot.rag.embedding_utils.get_search_client_for_doc_type")
    @patch("src.rag_chatbot.rag.embedding_utils.ensure_index_exists")
    @patch("src.rag_chatbot.rag.embedding_utils.generate_embeddings")
    @patch("src.rag_chatbot.rag.embedding_utils.extract_metadata")
    @patch("src.rag_chatbot.rag.embedding_utils.make_chunk_id")
    def test_returns_results_dict(
        self,
        mock_make_chunk_id,
        mock_extract_metadata,
        mock_generate_embeddings,
        mock_ensure_index,
        mock_get_client,
        sample_transcript_chunk,
    ):
        """Test that function returns correct results structure."""
        mock_make_chunk_id.return_value = "id-1"
        mock_extract_metadata.return_value = {}
        mock_generate_embeddings.return_value = [[0.1]]
        
        result_obj = Mock()
        result_obj.succeeded = True
        
        mock_client = Mock()
        mock_client._index_name = "test-index"
        mock_client.upload_documents.return_value = [result_obj]
        mock_get_client.return_value = mock_client

        result = process_and_store_chunks([sample_transcript_chunk])

        assert isinstance(result, dict)
        assert "transcript" in result
        assert len(result["transcript"]) == 1

    def test_raises_error_when_doctype_missing(self):
        """Test that ValueError is raised when docType is missing."""
        chunks = [
            {
                "source": "file.txt",
                "chunk_id": "1",
                "content": "content",
                # missing docType
            }
        ]
        
        with pytest.raises(ValueError, match="Chunk missing 'docType'"):
            process_and_store_chunks(chunks)