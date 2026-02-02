"""
Access control system for secure RAG.
Implements role-based access control (RBAC) for documents.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ClearanceLevel(Enum):
    """Security clearance levels."""
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


@dataclass
class User:
    """User with access control attributes."""
    user_id: str
    username: str
    clearance_level: ClearanceLevel
    roles: List[str]
    departments: List[str]


class AccessController:
    """
    Manages access control for documents based on user permissions.
    Implements RBAC and attribute-based access control (ABAC).
    """
    
    def __init__(self):
        """Initialize access controller."""
        self.audit_log = []
        logger.info("AccessController initialized")
    
    def can_access_document(
        self,
        user: User,
        document_metadata: Dict[str, Any]
    ) -> bool:
        """
        Check if user can access document.
        
        Args:
            user: User object with permissions
            document_metadata: Document metadata with access controls
            
        Returns:
            True if user has access, False otherwise
        """
        # Check clearance level
        doc_clearance = document_metadata.get("clearance_level")
        if doc_clearance:
            if isinstance(doc_clearance, str):
                doc_clearance = ClearanceLevel[doc_clearance.upper()]
            
            if user.clearance_level.value < doc_clearance.value:
                self._log_access_denied(
                    user.user_id,
                    document_metadata.get("document_id"),
                    "insufficient_clearance"
                )
                return False
        
        # Check required roles
        required_roles = document_metadata.get("required_roles", [])
        if required_roles:
            if not any(role in user.roles for role in required_roles):
                self._log_access_denied(
                    user.user_id,
                    document_metadata.get("document_id"),
                    "missing_role"
                )
                return False
        
        # Check department restrictions
        allowed_departments = document_metadata.get("allowed_departments", [])
        if allowed_departments:
            if not any(dept in user.departments for dept in allowed_departments):
                self._log_access_denied(
                    user.user_id,
                    document_metadata.get("document_id"),
                    "department_mismatch"
                )
                return False
        
        # Access granted
        self._log_access_granted(
            user.user_id,
            document_metadata.get("document_id")
        )
        
        return True
    
    def filter_documents(
        self,
        user: User,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter documents based on user permissions.
        
        Args:
            user: User object
            documents: List of documents with metadata
            
        Returns:
            Filtered list of accessible documents
        """
        accessible_docs = []
        
        for doc in documents:
            if self.can_access_document(user, doc.get("metadata", {})):
                accessible_docs.append(doc)
        
        logger.info(
            f"Filtered {len(documents)} documents to {len(accessible_docs)} "
            f"for user {user.username}"
        )
        
        return accessible_docs
    
    def get_filter_metadata(self, user: User) -> Dict[str, Any]:
        """
        Get metadata filter for vector store queries.
        
        Args:
            user: User object
            
        Returns:
            Dictionary with metadata filters
        """
        # This creates a filter that can be used directly with ChromaDB
        # Note: Actual implementation depends on your metadata structure
        return {
            "$or": [
                {"clearance_level": {"$lte": user.clearance_level.name}},
                {"clearance_level": None}
            ]
        }
    
    def _log_access_granted(self, user_id: str, document_id: Optional[str]):
        """Log successful access."""
        self.audit_log.append({
            "event": "access_granted",
            "user_id": user_id,
            "document_id": document_id,
            "timestamp": self._get_timestamp()
        })
    
    def _log_access_denied(
        self,
        user_id: str,
        document_id: Optional[str],
        reason: str
    ):
        """Log denied access."""
        self.audit_log.append({
            "event": "access_denied",
            "user_id": user_id,
            "document_id": document_id,
            "reason": reason,
            "timestamp": self._get_timestamp()
        })
        logger.warning(
            f"Access denied for user {user_id} to document {document_id}: {reason}"
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve audit log entries.
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of audit log entries
        """
        if user_id:
            filtered_log = [
                entry for entry in self.audit_log
                if entry["user_id"] == user_id
            ]
            return filtered_log[-limit:]
        
        return self.audit_log[-limit:]