#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security manager
Unified management interface integrating encryption, access control, and other security functionalities
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .encryption import EncryptionManager
from .access_control import AccessController, Permission, Role
from ..utils.config import Config


class SecurityManager:
    """Security manager"""

    def __init__(self, config: Config):
        """
        Initialize security manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.security_manager')

        # Initialize security components
        self.encryption_manager = EncryptionManager(config)
        self.access_controller = AccessController(config)

        # Security policy configuration
        self.security_policy = {
            'encrypt_sensitive_data': True,
            'require_authentication': True,
            'enable_audit_logging': True,
            'auto_lock_timeout_minutes': 30,
            'password_complexity_required': True,
            'max_login_attempts': 5
        }

        # Sensitive data types
        self.sensitive_data_types = {
            'memory_content',
            'user_preferences',
            'project_data',
            'conversation_history',
            'personal_information'
        }

        self.logger.info("Security manager initialization completed")

    def initialize_security(self, password: Optional[str] = None) -> bool:
        """
        Initialize security system

        Args:
            password: User password

        Returns:
            Whether initialization was successful
        """
        try:
            # Unlock encryption system
            if not self.encryption_manager.unlock(password):
                self.logger.error("Encryption system unlock failed")
                return False

            # Clean up expired sessions
            self.access_controller.cleanup_expired_sessions()

            self.logger.info("Security system initialization successful")
            return True

        except Exception as e:
            self.logger.error(f"Security system initialization failed: {e}")
            return False

    def authenticate_user(self, user_id: str, password: Optional[str] = None,
                         ip_address: Optional[str] = None,
                         user_agent: Optional[str] = None) -> Optional[str]:
        """
        User authentication

        Args:
            user_id: User ID
            password: Password
            ip_address: IP address
            user_agent: User agent

        Returns:
            Session ID, returns None on authentication failure
        """
        try:
            session_id = self.access_controller.authenticate(
                user_id, password, ip_address, user_agent
            )

            if session_id:
                self.logger.info(f"User authentication successful: {user_id}")
            else:
                self.logger.warning(f"User authentication failed: {user_id}")

            return session_id

        except Exception as e:
            self.logger.error(f"User authentication exception: {e}")
            return None

    def check_access(self, session_id: str, permission: Permission,
                    resource: Optional[str] = None) -> bool:
        """
        Check access permissions

        Args:
            session_id: Session ID
            permission: Required permission
            resource: Resource identifier

        Returns:
            Whether user has permission
        """
        return self.access_controller.check_permission(session_id, permission, resource)

    def require_access(self, session_id: str, permission: Permission,
                      resource: Optional[str] = None):
        """
        Require access permission

        Args:
            session_id: Session ID
            permission: Required permission
            resource: Resource identifier

        Raises:
            PermissionError: Insufficient permissions
        """
        self.access_controller.require_permission(session_id, permission, resource)

    def encrypt_sensitive_data(self, data: str, data_type: str) -> str:
        """
        Encrypt sensitive data

        Args:
            data: Data to encrypt
            data_type: Data type

        Returns:
            Encrypted data
        """
        try:
            if not self.security_policy['encrypt_sensitive_data']:
                return data

            if data_type in self.sensitive_data_types:
                return self.encryption_manager.encrypt_string(data, data_type)

            return data

        except Exception as e:
            self.logger.error(f"Data encryption failed: {e}")
            return data

    def decrypt_sensitive_data(self, encrypted_data: str, data_type: str) -> str:
        """
        Decrypt sensitive data

        Args:
            encrypted_data: Encrypted data
            data_type: Data type

        Returns:
            Decrypted data
        """
        try:
            if not self.security_policy['encrypt_sensitive_data']:
                return encrypted_data

            if data_type in self.sensitive_data_types:
                return self.encryption_manager.decrypt_string(encrypted_data, data_type)

            return encrypted_data

        except Exception as e:
            self.logger.error(f"Data decryption failed: {e}")
            return encrypted_data

    def secure_file_operation(self, file_path: Path, operation: str,
                             session_id: Optional[str] = None) -> bool:
        """
        Secure file operation

        Args:
            file_path: File path
            operation: Operation type (read/write/delete)
            session_id: Session ID

        Returns:
            Whether operation is allowed
        """
        try:
            # Check session permissions
            if session_id:
                if operation == 'read':
                    if not self.check_access(session_id, Permission.READ_MEMORY):
                        return False
                elif operation == 'write':
                    if not self.check_access(session_id, Permission.WRITE_MEMORY):
                        return False
                elif operation == 'delete':
                    if not self.check_access(session_id, Permission.DELETE_MEMORY):
                        return False

            # Check file path security
            if not self._is_safe_file_path(file_path):
                self.logger.warning(f"Unsafe file path: {file_path}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"File operation security check failed: {e}")
            return False

    def _is_safe_file_path(self, file_path: Path) -> bool:
        """Check if file path is safe"""
        try:
            # Parse absolute path
            abs_path = file_path.resolve()

            # Check if within allowed directories
            data_dir = Path(self.config.data_directory).resolve()

            try:
                abs_path.relative_to(data_dir)
                return True
            except ValueError:
                # Path not within data directory
                return False

        except Exception:
            return False

    def encrypt_file(self, file_path: Path, session_id: Optional[str] = None) -> Optional[Path]:
        """
        Encrypt file

        Args:
            file_path: File path
            session_id: Session ID

        Returns:
            Encrypted file path
        """
        try:
            # Check permissions
            if session_id and not self.check_access(session_id, Permission.WRITE_MEMORY):
                raise PermissionError("Insufficient permissions")

            # Check file path security
            if not self._is_safe_file_path(file_path):
                raise ValueError("Unsafe file path")

            return self.encryption_manager.encrypt_file(file_path)

        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            return None

    def decrypt_file(self, encrypted_file_path: Path, session_id: Optional[str] = None) -> Optional[Path]:
        """
        Decrypt file

        Args:
            encrypted_file_path: Encrypted file path
            session_id: Session ID

        Returns:
            Decrypted file path
        """
        try:
            # Check permissions
            if session_id and not self.check_access(session_id, Permission.READ_MEMORY):
                raise PermissionError("Insufficient permissions")

            # Check file path security
            if not self._is_safe_file_path(encrypted_file_path):
                raise ValueError("Unsafe file path")

            return self.encryption_manager.decrypt_file(encrypted_file_path)

        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            return None

    def change_password(self, session_id: str, old_password: str, new_password: str) -> bool:
        """
        Change password

        Args:
            session_id: Session ID
            old_password: Old password
            new_password: New password

        Returns:
            Whether change was successful
        """
        try:
            # Check permissions
            self.require_access(session_id, Permission.CONFIGURE_SYSTEM)

            # Verify password complexity
            if self.security_policy['password_complexity_required']:
                if not self._validate_password_complexity(new_password):
                    self.logger.warning("New password does not meet complexity requirements")
                    return False

            return self.encryption_manager.change_password(old_password, new_password)

        except Exception as e:
            self.logger.error(f"Password change failed: {e}")
            return False

    def _validate_password_complexity(self, password: str) -> bool:
        """Verify password complexity"""
        if len(password) < 8:
            return False

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        return sum([has_upper, has_lower, has_digit, has_special]) >= 3

    def logout_user(self, session_id: str) -> bool:
        """
        User logout

        Args:
            session_id: Session ID

        Returns:
            Whether logout was successful
        """
        return self.access_controller.logout(session_id)

    def get_security_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get security status

        Args:
            session_id: Session ID

        Returns:
            Security status information
        """
        try:
            # Check permissions
            self.require_access(session_id, Permission.VIEW_STATISTICS)

            encryption_info = self.encryption_manager.get_encryption_info()
            access_status = self.access_controller.get_security_status()

            return {
                'encryption': encryption_info,
                'access_control': access_status,
                'security_policy': self.security_policy.copy(),
                'sensitive_data_types': list(self.sensitive_data_types)
            }

        except Exception as e:
            self.logger.error(f"Failed to get security status: {e}")
            return {}

    def get_audit_log(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit log

        Args:
            session_id: Session ID
            limit: Log entry limit

        Returns:
            Audit log
        """
        try:
            # Check permissions
            self.require_access(session_id, Permission.ADMIN_ACCESS)

            return self.access_controller.get_access_log(limit)

        except Exception as e:
            self.logger.error(f"Failed to get audit log: {e}")
            return []

    def update_security_policy(self, session_id: str, **kwargs) -> bool:
        """
        Update security policy

        Args:
            session_id: Session ID
            **kwargs: Policy parameters

        Returns:
            Whether update was successful
        """
        try:
            # Check permissions
            self.require_access(session_id, Permission.CONFIGURE_SYSTEM)

            for key, value in kwargs.items():
                if key in self.security_policy:
                    self.security_policy[key] = value
                    self.logger.info(f"Security policy updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update security policy: {e}")
            return False

    def perform_security_check(self) -> Dict[str, Any]:
        """
        Perform security check

        Returns:
            Security check results
        """
        try:
            results = {
                'encryption_status': 'ok' if self.encryption_manager.is_unlocked() else 'locked',
                'active_sessions': len(self.access_controller.active_sessions),
                'failed_attempts': len(self.access_controller.failed_attempts),
                'locked_users': len(self.access_controller.locked_users),
                'issues': []
            }

            # Check encryption status
            if not self.encryption_manager.is_unlocked():
                results['issues'].append("Encryption system not unlocked")

            # Check expired sessions
            self.access_controller.cleanup_expired_sessions()

            # Check security policy
            if not self.security_policy['encrypt_sensitive_data']:
                results['issues'].append("Sensitive data encryption disabled")

            if not self.security_policy['require_authentication']:
                results['issues'].append("Authentication disabled")

            results['status'] = 'secure' if not results['issues'] else 'warning'

            return results

        except Exception as e:
            self.logger.error(f"Security check failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def shutdown_security(self):
        """Shutdown security system"""
        try:
            # Clear keys in memory
            self.encryption_manager.clear_keys()

            # Clean up active sessions
            self.access_controller.active_sessions.clear()

            self.logger.info("Security system shutdown completed")

        except Exception as e:
            self.logger.error(f"Failed to shutdown security system: {e}")