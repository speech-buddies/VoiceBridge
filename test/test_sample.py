def test_addition():
    """Simple test to verify pytest is working."""
    assert 1 + 1 == 2


def test_string_operations():
    """Test basic string operations."""
    text = "hello"
    assert text.upper() == "HELLO"
    assert len(text) == 5


def test_list_operations():
    """Test basic list operations."""
    my_list = [1, 2, 3]
    my_list.append(4)
    assert len(my_list) == 4
    assert 4 in my_list