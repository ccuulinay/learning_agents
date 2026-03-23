"""
Unit tests for mypackage.utils module.
"""

import pytest
from mypackage import utils


class TestIsValidEmail:
    """Tests for is_valid_email function."""
    
    def test_valid_email(self):
        assert utils.is_valid_email("user@example.com") is True
        assert utils.is_valid_email("user.name@example.co.uk") is True
        
    def test_invalid_email(self):
        assert utils.is_valid_email("invalid") is False
        assert utils.is_valid_email("@example.com") is False
        assert utils.is_valid_email("user@") is False
        
    def test_empty_string(self):
        assert utils.is_valid_email("") is False


class TestReverseString:
    """Tests for reverse_string function."""
    
    def test_reverse_simple(self):
        assert utils.reverse_string("hello") == "olleh"
        
    def test_reverse_empty(self):
        assert utils.reverse_string("") == ""
        
    def test_reverse_palindrome(self):
        assert utils.reverse_string("radar") == "radar"
        
    def test_reverse_with_spaces(self):
        assert utils.reverse_string("hello world") == "dlrow olleh"


class TestFindDuplicates:
    """Tests for find_duplicates function."""
    
    def test_with_duplicates(self):
        result = utils.find_duplicates([1, 2, 2, 3, 3, 3])
        assert sorted(result) == [2, 3]
        
    def test_no_duplicates(self):
        assert utils.find_duplicates([1, 2, 3]) == []
        
    def test_empty_list(self):
        assert utils.find_duplicates([]) == []
        
    def test_all_duplicates(self):
        result = utils.find_duplicates([1, 1, 1, 1])
        assert result == [1]


class TestSafeGet:
    """Tests for safe_get function."""
    
    def test_key_exists(self):
        data = {"name": "John", "age": 30}
        assert utils.safe_get(data, "name") == "John"
        
    def test_key_missing(self):
        data = {"name": "John"}
        assert utils.safe_get(data, "age") is None
        
    def test_key_missing_with_default(self):
        data = {"name": "John"}
        assert utils.safe_get(data, "age", 0) == 0
        
    def test_empty_dict(self):
        assert utils.safe_get({}, "key", "default") == "default"


class TestChunkList:
    """Tests for chunk_list function."""
    
    def test_chunk_evenly(self):
        result = utils.chunk_list([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]
        
    def test_chunk_unevenly(self):
        result = utils.chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]
        
    def test_chunk_size_larger_than_list(self):
        result = utils.chunk_list([1, 2], 5)
        assert result == [[1, 2]]
        
    def test_empty_list(self):
        assert utils.chunk_list([], 3) == []
        
    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            utils.chunk_list([1, 2, 3], 0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            utils.chunk_list([1, 2, 3], -1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
