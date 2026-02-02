"""
Prompt injection detection and prevention for secure RAG system.
"""

import logging
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class PromptGuard:
    """
    Guards against prompt injection and jailbreaking attempts.
    Analyzes queries for malicious patterns and blocks unsafe inputs.
    """
    
    # Known prompt injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction override
        r'ignore\s+(all\s+)?(previous|above|prior)\s+instructions',
        r'disregard\s+(all\s+)?(previous|above|prior)\s+instructions',
        r'forget\s+(all\s+)?(previous|above|prior)\s+instructions',
        
        # System/admin mode attempts
        r'(system|admin|root|developer)\s+mode',
        r'(system|admin)\s+override',
        r'(enable|activate|switch\s+to)\s+(admin|developer|debug)\s+mode',
        r'###\s*(system|admin)',
        
        # Role manipulation
        r'you\s+are\s+now',
        r'act\s+as\s+(if\s+)?(you|a)',
        r'pretend\s+(you|to\s+be)',
        r'simulate\s+(being|a)',
        
        # Context manipulation
        r'new\s+instructions',
        r'updated\s+instructions',
        r'revised\s+instructions',
        
        # Jailbreaking attempts
        r'(dan|do\s+anything\s+now)\s+mode',
        r'for\s+educational\s+purposes\s+only',
        r'in\s+a\s+hypothetical',
        r'let\'?s\s+play\s+a\s+game',
        
        # Data extraction attempts
        r'show\s+(me\s+)?(all|your)\s+(data|training|documents)',
        r'reveal\s+(your|the)\s+(prompt|instructions|data)',
        r'what\s+(are|is)\s+your\s+(instructions|prompt|training)',
        r'(print|display|output)\s+(your|the)\s+(system|prompt)',
    ]
    
    # Suspicious phrases
    SUSPICIOUS_PHRASES = [
        'ignore', 'bypass', 'override', 'disregard', 'forget',
        'pretend', 'simulate', 'act as', 'you are now',
        'system:', 'admin:', 'root:', '[INST]', '<|im_start|>',
        'developer mode', 'debug mode', 'jailbreak'
    ]
    
    def __init__(
        self,
        max_query_length: int = 1000,
        enable_pattern_detection: bool = True,
        enable_phrase_detection: bool = True,
        strictness: str = "medium"  # low, medium, high
    ):
        """
        Initialize prompt guard.
        
        Args:
            max_query_length: Maximum allowed query length
            enable_pattern_detection: Enable regex pattern detection
            enable_phrase_detection: Enable suspicious phrase detection
            strictness: Detection strictness level
        """
        self.max_query_length = max_query_length
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_phrase_detection = enable_phrase_detection
        self.strictness = strictness

        # data extraction patterns
        self.data_extraction_patterns = [
            r"(?i)show\s+(me\s+)?(all\s+)?(your\s+)?training\s+data",
            r"(?i)reveal\s+(your\s+)?system\s+prompt",
            r"(?i)what\s+are\s+your\s+instructions",
            r"(?i)print\s+(your\s+)?(entire\s+)?knowledge\s+base",
            r"(?i)dump\s+(all\s+)?(your\s+)?data",
            r'show\s+(me\s+)?(all|your)?\s*(data|training|documents)',
            r'reveal\s+(your|the)?\s*(prompt|instructions|data)',  # Make (your|the) optional with ?
            r'what\s+(are|is)\s+your\s+(instructions|prompt|training)',
            r'(print|display|output)\s+(your|the)?\s*(system|prompt)',
            r'reveal.*data',
            r'reveal.*prompt',
            r'reveal.*instructions',
            r'ignore.*instructions',
        ]
        
        # Set thresholds based on strictness
        if strictness == "low":
            self.phrase_threshold = 3
        elif strictness == "medium":
            self.phrase_threshold = 2
        else:  # high
            self.phrase_threshold = 1
        
        logger.info(f"PromptGuard initialized with strictness: {strictness}")
    
    def is_safe_query(self, query: str) -> bool:
        """
        Check if query is safe to process.
        
        Args:
            query: User query string
            
        Returns:
            True if query is safe, False otherwise
        """
        if not query:
            return True
        
        # Check length
        if len(query) > self.max_query_length:
            logger.warning(f"Query exceeds max length: {len(query)} > {self.max_query_length}")
            return False
        
        # Check for injection patterns
        if self.enable_pattern_detection:
            if self._detect_injection_patterns(query):
                logger.warning("Injection pattern detected in query")
                return False
        
        # Check for data extraction
        for pattern in self.data_extraction_patterns:
            if re.search(pattern, query):
                return False
        
        # Check for suspicious phrases
        if self.enable_phrase_detection:
            suspicious_count = self._count_suspicious_phrases(query)
            if suspicious_count >= self.phrase_threshold:
                logger.warning(f"Too many suspicious phrases: {suspicious_count}")
                return False
        
        return True
    
    def analyze_query(self, query: str) -> dict:
        """
        Detailed analysis of query safety.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "query": query,
            "is_safe": True,
            "length": len(query),
            "violations": [],
            "suspicious_phrases": [],
            "risk_score": 0.0
        }
        
        # Length check
        if len(query) > self.max_query_length:
            analysis["is_safe"] = False
            analysis["violations"].append("excessive_length")
            analysis["risk_score"] += 0.5
        
        # Check all patterns
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query):
                analysis["violations"].append(f"Injection pattern detected: {pattern}")
                analysis["risk_score"] += 0.3
        
        for pattern in self.data_extraction_patterns:
            if re.search(pattern, query):
                analysis["violations"].append(f"Data extraction attempt: {pattern}")
                analysis["risk_score"] += 0.4
        
        # Pattern detection
        if self.enable_pattern_detection:
            detected_patterns = self._get_detected_patterns(query)
            if detected_patterns:
                analysis["is_safe"] = False
                analysis["violations"].extend(detected_patterns)
                analysis["risk_score"] += 0.5 * len(detected_patterns)
        
        # Phrase detection
        if self.enable_phrase_detection:
            suspicious_phrases = self._get_suspicious_phrases(query)
            analysis["suspicious_phrases"] = suspicious_phrases
            
            if len(suspicious_phrases) >= self.phrase_threshold:
                analysis["is_safe"] = False
                analysis["violations"].append("suspicious_phrases")
                analysis["risk_score"] += 0.3 * len(suspicious_phrases)
        
        # Cap risk score at 1.0
        analysis["risk_score"] = min(analysis["risk_score"], 1.0)

        # Determine final safety based on violations
        if analysis["violations"] or analysis["risk_score"] > 0.5:
            analysis["is_safe"] = False
        
        return analysis
    
    def _detect_injection_patterns(self, query: str) -> bool:
        """Check if any injection patterns match."""
        query_lower = query.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _get_detected_patterns(self, query: str) -> List[str]:
        """Get list of detected injection patterns."""
        query_lower = query.lower()
        detected = []
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                detected.append(pattern[:50])  # Truncate for readability
        
        return detected
    
    def _count_suspicious_phrases(self, query: str) -> int:
        """Count number of suspicious phrases in query."""
        query_lower = query.lower()
        count = 0
        
        for phrase in self.SUSPICIOUS_PHRASES:
            if phrase.lower() in query_lower:
                count += 1
        
        return count
    
    def _get_suspicious_phrases(self, query: str) -> List[str]:
        """Get list of suspicious phrases found in query."""
        query_lower = query.lower()
        found = []
        
        for phrase in self.SUSPICIOUS_PHRASES:
            if phrase.lower() in query_lower:
                found.append(phrase)
        
        return found