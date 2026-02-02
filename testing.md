# Testing Guide

## Security Module Tests

### Run all security tests
```bash
pytest tests/test_security.py -v
```

### Run with coverage
```bash
pytest tests/test_security.py --cov=src.security --cov-report=term-missing
```

### Run specific test class
```bash
pytest tests/test_security.py::TestInputSanitizer -v
pytest tests/test_security.py::TestPromptGuard -v
pytest tests/test_security.py::TestAccessControl -v
```

### Quick manual verification
```bash
# Test imports
python -c "from src.security import InputSanitizer, PromptGuard, AccessController; print('✓ Imports OK')"

# Test sanitizer
python -c "from src.security import InputSanitizer; s = InputSanitizer(); r = s.sanitize('test@example.com'); print('✓ Sanitizer OK' if '[EMAIL]' in r.sanitized_text else '✗ Failed')"

# Test prompt guard
python -c "from src.security import PromptGuard; g = PromptGuard(); print('✓ Guard OK' if not g.is_safe_query('ignore instructions') else '✗ Failed')"
```

## Debugging

### Check module structure
```bash
python -c "import src.security; print(dir(src.security))"
```

### Test with Python shell
```bash
python
>>> from src.security import InputSanitizer
>>> sanitizer = InputSanitizer()
>>> result = sanitizer.sanitize("Contact me at test@example.com")
>>> print(result.sanitized_text)
>>> print(result.removed_patterns)
```