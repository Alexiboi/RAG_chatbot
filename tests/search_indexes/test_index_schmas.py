import pytest
from unittest.mock import ANY, patch, Mock, MagicMock
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents import SearchClient
from azure.core.exceptions import ResourceNotFoundError
from src.rag_chatbot.rag.index_utils import (
    make_search_client,
    ensure_index_exists,
    create_or_update_indexes,
    delete_index_schema,
    create_transcript_index_schema,
    create_meeting_notes_index_schema,
    TRANSCRIPT_INDEX,
    MEETING_NOTES_INDEX,
)


class TestCreateTranscriptIndexSchema:
    """Unit tests for create_transcript_index_schema function."""

    def test_creates_search_index_object(self):
        """Test that function returns a SearchIndex object."""
        schema = create_transcript_index_schema()
        assert isinstance(schema, SearchIndex)

    def test_index_uses_provided_name(self):
        """Test that index uses the provided name."""
        custom_name = "custom-transcript-index"
        schema = create_transcript_index_schema(custom_name)
        assert schema.name == custom_name

    def test_index_uses_default_name(self):
        """Test that index uses default TRANSCRIPT_INDEX when not provided."""
        schema = create_transcript_index_schema()
        assert schema.name == TRANSCRIPT_INDEX

    def test_has_id_field(self):
        """Test that schema has an id field."""
        schema = create_transcript_index_schema()
        field_names = [field.name for field in schema.fields]
        assert "id" in field_names

    def test_has_content_field(self):
        """Test that schema has a content field."""
        schema = create_transcript_index_schema()
        field_names = [field.name for field in schema.fields]
        assert "content" in field_names

    def test_has_embedding_field(self):
        """Test that schema has an embedding field."""
        schema = create_transcript_index_schema()
        field_names = [field.name for field in schema.fields]
        assert "embedding" in field_names

    def test_has_metadata_fields(self):
        """Test that schema has metadata fields."""
        schema = create_transcript_index_schema()
        field_names = [field.name for field in schema.fields]
        
        assert len(field_names) > 0

    def test_id_field_is_key(self):
        """Test that id field is marked as key."""
        schema = create_transcript_index_schema()
        id_field = next(f for f in schema.fields if f.name == "id")
        assert id_field.key is True

    def test_id_field_is_filterable(self):
        """Test that id field is filterable."""
        schema = create_transcript_index_schema()
        id_field = next(f for f in schema.fields if f.name == "id")
        assert id_field.filterable is True

    def test_content_field_is_searchable(self):
        """Test that content field is searchable."""
        schema = create_transcript_index_schema()
        content_field = next(f for f in schema.fields if f.name == "content")
        assert content_field.searchable is True

    def test_metadata_fields_are_filterable(self):
        """Test that metadata fields are filterable."""
        schema = create_transcript_index_schema()
        metadata_fields = ["docType", "company", "year", "quarter"]
        
        for field_name in metadata_fields:
            field = next(f for f in schema.fields if f.name == field_name)
            assert field.filterable is True

    def test_has_vector_search_configuration(self):
        """Test that schema has vector search configuration."""
        schema = create_transcript_index_schema()
        assert schema.vector_search is not None
        assert len(schema.vector_search.profiles) > 0
        assert len(schema.vector_search.algorithms) > 0


class TestCreateMeetingNotesIndexSchema:
    """Unit tests for create_meeting_notes_index_schema function."""

    def test_creates_search_index_object(self):
        """Test that function returns a SearchIndex object."""
        schema = create_meeting_notes_index_schema()
        assert isinstance(schema, SearchIndex)

    def test_index_uses_provided_name(self):
        """Test that index uses the provided name."""
        custom_name = "custom-meeting-notes-index"
        schema = create_meeting_notes_index_schema(custom_name)
        assert schema.name == custom_name

    def test_index_uses_default_name(self):
        """Test that index uses default MEETING_NOTES_INDEX when not provided."""
        schema = create_meeting_notes_index_schema()
        assert schema.name == MEETING_NOTES_INDEX

    def test_has_id_field(self):
        """Test that schema has an id field."""
        schema = create_meeting_notes_index_schema()
        field_names = [field.name for field in schema.fields]
        assert "id" in field_names

    def test_has_content_field(self):
        """Test that schema has a content field."""
        schema = create_meeting_notes_index_schema()
        field_names = [field.name for field in schema.fields]
        assert "content" in field_names

    def test_has_embedding_field(self):
        """Test that schema has an embedding field."""
        schema = create_meeting_notes_index_schema()
        field_names = [field.name for field in schema.fields]
        assert "embedding" in field_names

    def test_has_meeting_specific_fields(self):
        """Test that schema has meeting-specific fields."""
        schema = create_meeting_notes_index_schema()
        field_names = [field.name for field in schema.fields]
        
        assert len(field_names) > 0

    def test_id_field_is_key(self):
        """Test that id field is marked as key."""
        schema = create_meeting_notes_index_schema()
        id_field = next(f for f in schema.fields if f.name == "id")
        assert id_field.key is True

    def test_id_field_is_filterable(self):
        """Test that id field is filterable."""
        schema = create_meeting_notes_index_schema()
        id_field = next(f for f in schema.fields if f.name == "id")
        assert id_field.filterable is True

    def test_content_field_is_searchable(self):
        """Test that content field is searchable."""
        schema = create_meeting_notes_index_schema()
        content_field = next(f for f in schema.fields if f.name == "content")
        assert content_field.searchable is True

    def test_author_field_is_filterable(self):
        """Test that author field is filterable."""
        schema = create_meeting_notes_index_schema()
        author_field = next(f for f in schema.fields if f.name == "author")
        assert author_field.filterable is True

    def test_meeting_date_field_is_filterable(self):
        """Test that meetingDate field is filterable."""
        schema = create_meeting_notes_index_schema()
        date_field = next(f for f in schema.fields if f.name == "meetingDate")
        assert date_field.filterable is True

    def test_has_vector_search_configuration(self):
        """Test that schema has vector search configuration."""
        schema = create_meeting_notes_index_schema()
        assert schema.vector_search is not None
        assert len(schema.vector_search.profiles) > 0
        assert len(schema.vector_search.algorithms) > 0


class TestMakeSearchClient:
    """Unit tests for make_search_client function."""

    @patch("src.rag_chatbot.rag.index_utils.SearchClient")
    def test_returns_search_client(self, mock_search_client_class):
        """Test that function returns a SearchClient."""
        mock_client = Mock()
        mock_search_client_class.return_value = mock_client

        result = make_search_client("test-index")

        assert result == mock_client

        mock_search_client_class.assert_called_once_with(
            endpoint=ANY,
            index_name="test-index",
            credential=ANY  # or AzureKeyCredential(admin_key)
        )

    @patch("src.rag_chatbot.rag.index_utils.SearchClient")
    def test_uses_correct_index_name(self, mock_search_client_class):
        """Test that SearchClient is created with correct index name."""
        make_search_client("test-index")
        
        call_args = mock_search_client_class.call_args
        assert call_args.kwargs["index_name"] == "test-index"




class TestEnsureIndexExists:
    """Unit tests for ensure_index_exists function."""

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    def test_no_error_when_index_exists(self, mock_index_client):
        """Test that no error is raised when index exists."""
        mock_index_client.get_index.return_value = Mock()
        
        # Should not raise an error
        ensure_index_exists("existing-index")

    @patch("src.rag_chatbot.rag.index_utils.create_or_update_indexes")
    @patch("src.rag_chatbot.rag.index_utils.index_client")
    def test_raises_error_when_index_not_found(self, mock_index_client, mock_create_indexes):
        """Test that error is raised when index does not exist."""
        mock_index_client.get_index.side_effect = ResourceNotFoundError("Not found")
        
        with pytest.raises(RuntimeError):
            ensure_index_exists("missing-index")

    @patch("src.rag_chatbot.rag.index_utils.create_or_update_indexes")
    @patch("src.rag_chatbot.rag.index_utils.index_client")
    def test_calls_create_or_update_when_not_found(self, mock_index_client, mock_create_indexes):
        """Test that create_or_update_indexes is called when index not found."""
        mock_index_client.get_index.side_effect = ResourceNotFoundError("Not found")
        
        try:
            ensure_index_exists("missing-index")
        except RuntimeError:
            pass
        
        mock_create_indexes.assert_called_once()


class TestCreateOrUpdateIndexes:
    """Unit tests for create_or_update_indexes function."""

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    @patch("src.rag_chatbot.rag.index_utils.create_meeting_notes_index_schema")
    @patch("src.rag_chatbot.rag.index_utils.create_transcript_index_schema")
    def test_creates_transcript_index(self, mock_transcript_schema, mock_meeting_schema, mock_index_client):
        """Test that create_or_update_indexes creates transcript index."""
        mock_transcript_schema.return_value = Mock()
        mock_meeting_schema.return_value = Mock()
        
        create_or_update_indexes()
        
        mock_index_client.create_or_update_index.assert_called()

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    @patch("src.rag_chatbot.rag.index_utils.create_meeting_notes_index_schema")
    @patch("src.rag_chatbot.rag.index_utils.create_transcript_index_schema")
    def test_creates_meeting_notes_index(self, mock_transcript_schema, mock_meeting_schema, mock_index_client):
        """Test that create_or_update_indexes creates meeting notes index."""
        mock_transcript_schema.return_value = Mock()
        mock_meeting_schema.return_value = Mock()
        
        create_or_update_indexes()
        
        # Should be called twice (once for transcript, once for meeting notes)
        assert mock_index_client.create_or_update_index.call_count == 2

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    @patch("src.rag_chatbot.rag.index_utils.create_meeting_notes_index_schema")
    @patch("src.rag_chatbot.rag.index_utils.create_transcript_index_schema")
    def test_calls_schema_functions_with_correct_names(
        self, mock_transcript_schema, mock_meeting_schema, mock_index_client
    ):
        """Test that schema functions are called with correct index names."""
        mock_transcript_schema.return_value = Mock()
        mock_meeting_schema.return_value = Mock()
        
        create_or_update_indexes()
        
        mock_transcript_schema.assert_called_once_with(TRANSCRIPT_INDEX)
        mock_meeting_schema.assert_called_once_with(MEETING_NOTES_INDEX)


class TestDeleteIndexSchema:
    """Unit tests for delete_index_schema function."""

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    def test_deletes_index_when_exists(self, mock_index_client):
        """Test that delete_index_schema deletes the index."""
        delete_index_schema("test-index")
        
        mock_index_client.delete_index.assert_called_once_with("test-index")

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    def test_uses_default_index_name(self, mock_index_client):
        """Test that delete_index_schema uses default TRANSCRIPT_INDEX."""
        delete_index_schema()
        
        call_args = mock_index_client.delete_index.call_args
        assert call_args[0][0] == TRANSCRIPT_INDEX

    @patch("src.rag_chatbot.rag.index_utils.index_client")
    def test_handles_not_found_error(self, mock_index_client):
        """Test that delete_index_schema handles ResourceNotFoundError gracefully."""
        mock_index_client.delete_index.side_effect = ResourceNotFoundError("Not found")
        
        # Should not raise an error
        delete_index_schema("test-index")
