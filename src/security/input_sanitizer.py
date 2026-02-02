"""
Input sanitization module for secure RAG system.
Removes PII, malicious content, and applies security filters.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of sanitization operation."""
    sanitized_text: str
    removed_patterns: List[str]
    is_safe: bool
    warnings: List[str]


class InputSanitizer:
    """
    Sanitizes inputs to prevent security vulnerabilities.
    Removes PII, detects malicious patterns, and filters unsafe content.
    """
    
    # PII patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    IP_ADDRESS_PATTERN = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    
    # Malicious patterns (lowercase for case-insensitive matching)
    MALICIOUS_PATTERNS = [
        r'ignore\s+previous\s+instructions',
        r'system\s+override',
        r'admin\s+mode',
        r'bypass\s+security',
        r'<script[^>]*>',  # XSS attempts
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
        r'\'\s*or\s*\'1\'\s*=\s*\'1',  # SQL injection
        r'union\s+select',  # SQL injection
        r'drop\s+table',  # SQL injection
    ]
    
    def __init__(
        self,
        remove_emails: bool = True,
        remove_phones: bool = True,
        remove_ssn: bool = True,
        remove_credit_cards: bool = True,
        remove_ip_addresses: bool = False,
        detect_malicious: bool = True
    ):
        """
        Initialize input sanitizer.
        
        Args:
            remove_emails: Remove email addresses
            remove_phones: Remove phone numbers
            remove_ssn: Remove SSNs
            remove_credit_cards: Remove credit card numbers
            remove_ip_addresses: Remove IP addresses
            detect_malicious: Detect malicious patterns
        """
        self.remove_emails = remove_emails
        self.remove_phones = remove_phones
        self.remove_ssn = remove_ssn
        self.remove_credit_cards = remove_credit_cards
        self.remove_ip_addresses = remove_ip_addresses
        self.detect_malicious = detect_malicious

        self.sql_patterns = [
            r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\s+",
            r"(?i)(UNION|JOIN)\s+",
            r"(?i)(\bOR\b|\bAND\b)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?",
            r"['\"];?\s*(DROP|DELETE|INSERT)",
        ]


        
        logger.info("InputSanitizer initialized")
    
    def _detect_sql_injection(self, text: str) -> bool:
        """Detect SQL injection attempts."""
        for pattern in self.sql_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def sanitize(self, text: str) -> SanitizationResult:
        """
        Sanitize input text.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            SanitizationResult with sanitized text and metadata
        """
        if not text:
            return SanitizationResult(
                sanitized_text="",
                removed_patterns=[],
                is_safe=True,
                warnings=[]
            )
        
        sanitized = text
        removed_patterns = []
        warnings = []
        is_safe = True
        
        # Check for malicious patterns first
        if self.detect_malicious:
            malicious_found = self._detect_malicious_patterns(sanitized.lower())
            if malicious_found:
                is_safe = False
                warnings.append(f"Malicious patterns detected: {', '.join(malicious_found)}")
                removed_patterns.extend(malicious_found)
        
        # SQL injection check
        if self._detect_sql_injection(text):
            return SanitizationResult(
                sanitized_text=text,
                removed_patterns=[],
                is_safe=False,
                warnings=["Potential SQL injection detected"]
            )
        
        # Remove PII
        if self.remove_emails:
            sanitized, count = self._remove_pattern(sanitized, self.EMAIL_PATTERN, "[EMAIL]")
            if count > 0:
                removed_patterns.append(f"emails({count})")
        
        if self.remove_phones:
            sanitized, count = self._remove_pattern(sanitized, self.PHONE_PATTERN, "[PHONE]")
            if count > 0:
                removed_patterns.append(f"phones({count})")
        
        if self.remove_ssn:
            sanitized, count = self._remove_pattern(sanitized, self.SSN_PATTERN, "[SSN]")
            if count > 0:
                removed_patterns.append(f"ssn({count})")
        
        if self.remove_credit_cards:
            sanitized, count = self._remove_pattern(sanitized, self.CREDIT_CARD_PATTERN, "[CREDIT_CARD]")
            if count > 0:
                removed_patterns.append(f"credit_cards({count})")
        
        if self.remove_ip_addresses:
            sanitized, count = self._remove_pattern(sanitized, self.IP_ADDRESS_PATTERN, "[IP_ADDRESS]")
            if count > 0:
                removed_patterns.append(f"ip_addresses({count})")
        
        return SanitizationResult(
            sanitized_text=sanitized.strip(),
            removed_patterns=removed_patterns,
            is_safe=is_safe,
            warnings=warnings
        )
    
    def _remove_pattern(
        self,
        text: str,
        pattern: str,
        replacement: str
    ) -> Tuple[str, int]:
        """
        Remove pattern from text and return count.
        
        Args:
            text: Input text
            pattern: Regex pattern
            replacement: Replacement string
            
        Returns:
            Tuple of (sanitized_text, count_removed)
        """
        matches = re.findall(pattern, text)
        sanitized = re.sub(pattern, replacement, text)
        return sanitized, len(matches)
    
    def _detect_malicious_patterns(self, text: str) -> List[str]:
        """
        Detect malicious patterns in text.
        
        Args:
            text: Input text (should be lowercased)
            
        Returns:
            List of detected pattern descriptions
        """
        detected = []
        
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(pattern)
        
        return detected
    
    def sanitize_metadata(self, metadata: Dict) -> Dict:
        """
        Sanitize metadata dictionary.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, str):
                result = self.sanitize(value)
                sanitized_metadata[key] = result.sanitized_text
            else:
                sanitized_metadata[key] = value
        
        return sanitized_metadata
