#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Access controller
Provides role-based access control and permission management functionality
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from ..utils.config import Config


class Permission(Enum):
    """Permission enumeration"""
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    DELETE_MEMORY = "delete_memory"
    MANAGE_PROJECTS = "manage_projects"
    CONFIGURE_SYSTEM = "configure_system"
    VIEW_STATISTICS = "view_statistics"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    MANAGE_USERS = "manage_users"
    ADMIN_ACCESS = "admin_access"


class Role(Enum):
    """Role enumeration"""
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"


@dataclass
class AccessSession:
    """Access session"""
    session_id: str
    user_id: str
    role: Role
    permissions: Set[Permission]
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class AccessAttempt:
    """Access attempt record"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    success: bool
    ip_address: Optional[str] = None
    reason: Optional[str] = None


class AccessController:
    """Access controller"""

    def __init__(self, config: Config):
        """
        Initialize access controller

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.access_controller')

        # Access control configuration
        self.access_config = {
            'session_timeout_minutes': 480,  # 8 hours
            'max_failed_attempts': 5,
            'lockout_duration_minutes': 30,
            'require_authentication': True,
            'enable_audit_log': True,
            'max_concurrent_sessions': 3
        }

        # Role permission mapping
        self.role_permissions = {
            Role.GUEST: {
                Permission.READ_MEMORY,
                Permission.VIEW_STATISTICS
            },
            Role.USER: {
                Permission.READ_MEMORY,
                Permission.WRITE_MEMORY,
                Permission.VIEW_STATISTICS,
                Permission.EXPORT_DATA
            },
            Role.POWER_USER: {
                Permission.READ_MEMORY,
                Permission.WRITE_MEMORY,
                Permission.DELETE_MEMORY,
                Permission.MANAGE_PROJECTS,
                Permission.VIEW_STATISTICS,
                Permission.EXPORT_DATA,
                Permission.IMPORT_DATA
            },
            Role.ADMIN: set(Permission)  # All permissions
        }

        # Runtime state
        self.active_sessions: Dict[str, AccessSession] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_users: Dict[str, datetime] = {}
        self.access_log: List[AccessAttempt] = []

        # Default user (for single-user mode)
        self.default_user_id = "default_user"
        self.default_role = Role.ADMIN

    def authenticate(self, user_id: str, password: Optional[str] = None,
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
            # Check if authentication is required
            if not self.access_config['require_authentication']:
                return self._create_session(
                    self.default_user_id,
                    self.default_role,
                    ip_address,
                    user_agent
                )

            # Check if user is locked
            if self._is_user_locked(user_id):
                self._log_access_attempt(
                    user_id, "authenticate", "system",
                    False, ip_address, "User locked"
                )
                return None

            # Verify password
            if not self._verify_password(user_id, password):
                self._record_failed_attempt(user_id)
                self._log_access_attempt(
                    user_id, "authenticate", "system",
                    False, ip_address, "Incorrect password"
                )
                return None

            # Clear failed attempt records
            self._clear_failed_attempts(user_id)

            # Get user role
            role = self._get_user_role(user_id)

            # Create session
            session_id = self._create_session(user_id, role, ip_address, user_agent)

            self._log_access_attempt(
                user_id, "authenticate", "system",
                True, ip_address, "Authentication successful"
            )

            return session_id

        except Exception as e:
            self.logger.error(f"User authentication failed: {e}")
            return None

    def _verify_password(self, user_id: str, password: Optional[str]) -> bool:
        """Verify password"""
        if not password:
            return False

        # In production, password hash should be obtained from secure storage
        # Currently using simple default password verification
        expected_password = self._get_user_password(user_id)
        return self._hash_password(password) == self._hash_password(expected_password)

    def _get_user_password(self, user_id: str) -> str:
        """Get user password"""
        # Simplified implementation: use environment variable or default password
        import os
        return os.environ.get('MYJAVIS_PASSWORD', 'hibro123')

    def _hash_password(self, password: str) -> str:
        """Password hash"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _get_user_role(self, user_id: str) -> Role:
        """Get user role"""
        # Simplified implementation: default to admin role
        return Role.ADMIN

    def _is_user_locked(self, user_id: str) -> bool:
        """Check if user is locked"""
        if user_id not in self.locked_users:
            return False

        lock_time = self.locked_users[user_id]
        unlock_time = lock_time + timedelta(minutes=self.access_config['lockout_duration_minutes'])

        if datetime.now() >= unlock_time:
            # Unlock user
            del self.locked_users[user_id]
            return False

        return True

    def _record_failed_attempt(self, user_id: str):
        """Record failed attempt"""
        now = datetime.now()

        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []

        self.failed_attempts[user_id].append(now)

        # Clean up expired failure records
        cutoff_time = now - timedelta(minutes=self.access_config['lockout_duration_minutes'])
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt >= cutoff_time
        ]

        # Check if user needs to be locked
        if len(self.failed_attempts[user_id]) >= self.access_config['max_failed_attempts']:
            self.locked_users[user_id] = now
            self.logger.warning(f"User {user_id} locked due to multiple failed attempts")

    def _clear_failed_attempts(self, user_id: str):
        """Clear failed attempt records"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]

    def _create_session(self, user_id: str, role: Role,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> str:
        """Create access session"""
        # Check concurrent session limit
        user_sessions = [
            session for session in self.active_sessions.values()
            if session.user_id == user_id
        ]

        if len(user_sessions) >= self.access_config['max_concurrent_sessions']:
            # Remove oldest session
            oldest_session = min(user_sessions, key=lambda s: s.last_activity)
            del self.active_sessions[oldest_session.session_id]

        # Generate session ID
        session_id = secrets.token_urlsafe(32)

        # Create session
        session = AccessSession(
            session_id=session_id,
            user_id=user_id,
            role=role,
            permissions=self.role_permissions[role].copy(),
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.active_sessions[session_id] = session

        self.logger.info(f"Created session: {session_id} for {user_id}")
        return session_id

    def validate_session(self, session_id: str) -> Optional[AccessSession]:
        """
        Validate session

        Args:
            session_id: Session ID

        Returns:
            Session object, returns None if invalid
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # Check if session has expired
        timeout = timedelta(minutes=self.access_config['session_timeout_minutes'])
        if datetime.now() - session.last_activity > timeout:
            del self.active_sessions[session_id]
            self.logger.info(f"Session expired: {session_id}")
            return None

        # Update last activity time
        session.last_activity = datetime.now()
        return session

    def check_permission(self, session_id: str, permission: Permission,
                        resource: Optional[str] = None) -> bool:
        """
        Check permission

        Args:
            session_id: Session ID
            permission: Required permission
            resource: Resource identifier

        Returns:
            Whether user has permission
        """
        try:
            session = self.validate_session(session_id)
            if not session:
                self._log_access_attempt(
                    "unknown", permission.value, resource or "unknown",
                    False, None, "Invalid session"
                )
                return False

            has_permission = permission in session.permissions

            self._log_access_attempt(
                session.user_id, permission.value, resource or "unknown",
                has_permission, session.ip_address,
                "Permission check" if has_permission else "Insufficient permissions"
            )

            return has_permission

        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            return False

    def require_permission(self, session_id: str, permission: Permission,
                          resource: Optional[str] = None):
        """
        Require permission (for decorator use)

        Args:
            session_id: Session ID
            permission: Required permission
            resource: Resource identifier

        Raises:
            PermissionError: Insufficient permissions
        """
        if not self.check_permission(session_id, permission, resource):
            raise PermissionError(f"Insufficient permissions: {permission.value}")

    def logout(self, session_id: str) -> bool:
        """
        User logout

        Args:
            session_id: Session ID

        Returns:
            Whether logout was successful
        """
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                del self.active_sessions[session_id]

                self._log_access_attempt(
                    session.user_id, "logout", "system",
                    True, session.ip_address, "User logout"
                )

                self.logger.info(f"User logged out: {session.user_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"User logout failed: {e}")
            return False

    def _log_access_attempt(self, user_id: str, action: str, resource: str,
                           success: bool, ip_address: Optional[str] = None,
                           reason: Optional[str] = None):
        """Record access attempt"""
        if not self.access_config['enable_audit_log']:
            return

        attempt = AccessAttempt(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            ip_address=ip_address,
            reason=reason
        )

        self.access_log.append(attempt)

        # Limit log size
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get active session list"""
        sessions = []
        for session in self.active_sessions.values():
            sessions.append({
                'session_id': session.session_id,
                'user_id': session.user_id,
                'role': session.role.value,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'ip_address': session.ip_address
            })
        return sessions

    def get_access_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access log"""
        log_entries = []
        for attempt in self.access_log[-limit:]:
            log_entries.append({
                'timestamp': attempt.timestamp.isoformat(),
                'user_id': attempt.user_id,
                'action': attempt.action,
                'resource': attempt.resource,
                'success': attempt.success,
                'ip_address': attempt.ip_address,
                'reason': attempt.reason
            })
        return log_entries

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        timeout = timedelta(minutes=self.access_config['session_timeout_minutes'])
        now = datetime.now()

        for session_id, session in self.active_sessions.items():
            if now - session.last_activity > timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def update_access_config(self, **kwargs) -> bool:
        """
        Update access control configuration

        Args:
            **kwargs: Configuration parameters

        Returns:
            Whether update was successful
        """
        try:
            for key, value in kwargs.items():
                if key in self.access_config:
                    self.access_config[key] = value
                    self.logger.info(f"Access control configuration updated: {key} = {value}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update access control configuration: {e}")
            return False

    # ==================== Stage 4 Enterprise Security Enhancement Features ====================

    def enable_multi_factor_auth(self, user_id: str, mfa_secret: str) -> bool:
        """
        Enable multi-factor authentication

        Args:
            user_id: User ID
            mfa_secret: MFA secret key

        Returns:
            Whether enablement was successful
        """
        try:
            # In production, this should be stored in a secure database
            if not hasattr(self, '_mfa_secrets'):
                self._mfa_secrets = {}

            self._mfa_secrets[user_id] = mfa_secret
            self.logger.info(f"Multi-factor authentication enabled for user {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to enable multi-factor authentication: {e}")
            return False

    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """
        Verify MFA token

        Args:
            user_id: User ID
            token: MFA token

        Returns:
            Whether verification was successful
        """
        try:
            if not hasattr(self, '_mfa_secrets') or user_id not in self._mfa_secrets:
                return False

            # Simplified TOTP verification implementation
            import time
            import hmac
            import struct

            secret = self._mfa_secrets[user_id]
            current_time = int(time.time()) // 30  # 30-second time window

            # Allow tokens from one time window before or after
            for time_offset in [-1, 0, 1]:
                test_time = current_time + time_offset
                expected_token = self._generate_totp(secret, test_time)
                if token == expected_token:
                    return True

            return False

        except Exception as e:
            self.logger.error(f"MFA token verification failed: {e}")
            return False

    def _generate_totp(self, secret: str, time_counter: int) -> str:
        """Generate TOTP token"""
        import hmac
        import hashlib
        import struct
        import base64

        # Convert secret to bytes
        key = base64.b32decode(secret.upper() + '=' * (8 - len(secret) % 8))

        # Convert time counter to 8-byte big-endian
        time_bytes = struct.pack('>Q', time_counter)

        # HMAC-SHA1
        hmac_digest = hmac.new(key, time_bytes, hashlib.sha1).digest()

        # Dynamic truncation
        offset = hmac_digest[-1] & 0x0f
        truncated = struct.unpack('>I', hmac_digest[offset:offset + 4])[0]
        truncated &= 0x7fffffff

        # Generate 6-digit number
        return str(truncated % 1000000).zfill(6)

    def authenticate_with_mfa(self, user_id: str, password: str, mfa_token: str,
                             ip_address: Optional[str] = None,
                             user_agent: Optional[str] = None) -> Optional[str]:
        """
        User authentication with MFA

        Args:
            user_id: User ID
            password: Password
            mfa_token: MFA token
            ip_address: IP address
            user_agent: User agent

        Returns:
            Session ID, returns None on authentication failure
        """
        try:
            # First perform basic authentication
            if not self._verify_password(user_id, password):
                self._record_failed_attempt(user_id)
                self._log_access_attempt(
                    user_id, "authenticate_mfa", "system",
                    False, ip_address, "Incorrect password"
                )
                return None

            # Verify MFA token
            if not self.verify_mfa_token(user_id, mfa_token):
                self._record_failed_attempt(user_id)
                self._log_access_attempt(
                    user_id, "authenticate_mfa", "system",
                    False, ip_address, "Invalid MFA token"
                )
                return None

            # Perform risk assessment
            risk_score = self._assess_authentication_risk(user_id, ip_address, user_agent)

            if risk_score > 0.8:  # High risk
                self._log_access_attempt(
                    user_id, "authenticate_mfa", "system",
                    False, ip_address, f"High-risk authentication blocked, risk score: {risk_score:.2f}"
                )
                return None

            # Create session
            role = self._get_user_role(user_id)
            session_id = self._create_session(user_id, role, ip_address, user_agent)

            # Log successful MFA authentication
            self._log_access_attempt(
                user_id, "authenticate_mfa", "system",
                True, ip_address, f"MFA authentication successful, risk score: {risk_score:.2f}"
            )

            return session_id

        except Exception as e:
            self.logger.error(f"MFA authentication failed: {e}")
            return None

    def _assess_authentication_risk(self, user_id: str, ip_address: Optional[str],
                                   user_agent: Optional[str]) -> float:
        """
        Assess authentication risk

        Args:
            user_id: User ID
            ip_address: IP address
            user_agent: User agent

        Returns:
            Risk score (0.0-1.0)
        """
        risk_score = 0.0

        try:
            # Check IP address history
            if ip_address:
                user_ips = self._get_user_ip_history(user_id)
                if ip_address not in user_ips:
                    risk_score += 0.3  # New IP address

                # Check IP geolocation change (simplified implementation)
                if self._is_suspicious_ip_location(ip_address, user_ips):
                    risk_score += 0.4

            # Check user agent change
            if user_agent:
                user_agents = self._get_user_agent_history(user_id)
                if user_agent not in user_agents:
                    risk_score += 0.2  # New user agent

            # Check authentication time pattern
            if self._is_unusual_login_time(user_id):
                risk_score += 0.2

            # Check recent failed attempts
            if user_id in self.failed_attempts and len(self.failed_attempts[user_id]) > 0:
                risk_score += 0.1

            return min(risk_score, 1.0)

        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return 0.5  # Default medium risk

    def _get_user_ip_history(self, user_id: str) -> Set[str]:
        """Get user IP history"""
        ips = set()
        for attempt in self.access_log:
            if attempt.user_id == user_id and attempt.success and attempt.ip_address:
                ips.add(attempt.ip_address)
        return ips

    def _get_user_agent_history(self, user_id: str) -> Set[str]:
        """Get user agent history"""
        # Get user agent history from active sessions
        agents = set()
        for session in self.active_sessions.values():
            if session.user_id == user_id and session.user_agent:
                agents.add(session.user_agent)
        return agents

    def _is_suspicious_ip_location(self, ip_address: str, known_ips: Set[str]) -> bool:
        """Check if IP location is suspicious"""
        # Simplified implementation: check IP address subnet
        if not known_ips:
            return False

        current_subnet = '.'.join(ip_address.split('.')[:3])
        for known_ip in known_ips:
            known_subnet = '.'.join(known_ip.split('.')[:3])
            if current_subnet == known_subnet:
                return False

        return True  # Different subnet, may be suspicious

    def _is_unusual_login_time(self, user_id: str) -> bool:
        """Check if login time is unusual"""
        import datetime

        current_hour = datetime.datetime.now().hour

        # Get user's historical login times
        login_hours = []
        for attempt in self.access_log:
            if (attempt.user_id == user_id and
                attempt.action == "authenticate" and
                attempt.success):
                login_hours.append(attempt.timestamp.hour)

        if not login_hours:
            return False

        # Simple anomaly detection: if current time differs significantly from historical login times
        avg_hour = sum(login_hours) / len(login_hours)
        time_diff = abs(current_hour - avg_hour)

        # Consider 24-hour cycle
        time_diff = min(time_diff, 24 - time_diff)

        return time_diff > 6  # More than 6 hours difference is considered unusual

    def add_ip_whitelist(self, ip_addresses: List[str]) -> bool:
        """
        Add IP whitelist

        Args:
            ip_addresses: List of IP addresses

        Returns:
            Whether addition was successful
        """
        try:
            if not hasattr(self, '_ip_whitelist'):
                self._ip_whitelist = set()

            for ip in ip_addresses:
                self._ip_whitelist.add(ip)

            self.logger.info(f"Added IP whitelist: {ip_addresses}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add IP whitelist: {e}")
            return False

    def add_ip_blacklist(self, ip_addresses: List[str]) -> bool:
        """
        Add IP blacklist

        Args:
            ip_addresses: List of IP addresses

        Returns:
            Whether addition was successful
        """
        try:
            if not hasattr(self, '_ip_blacklist'):
                self._ip_blacklist = set()

            for ip in ip_addresses:
                self._ip_blacklist.add(ip)

            self.logger.info(f"Added IP blacklist: {ip_addresses}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add IP blacklist: {e}")
            return False

    def is_ip_allowed(self, ip_address: str) -> bool:
        """
        Check if IP is allowed

        Args:
            ip_address: IP address

        Returns:
            Whether access is allowed
        """
        # Check blacklist
        if hasattr(self, '_ip_blacklist') and ip_address in self._ip_blacklist:
            return False

        # Check whitelist (if whitelist exists, only allow whitelisted IPs)
        if hasattr(self, '_ip_whitelist') and self._ip_whitelist:
            return ip_address in self._ip_whitelist

        return True

    def detect_session_anomalies(self, session_id: str) -> Dict[str, Any]:
        """
        Detect session anomalies

        Args:
            session_id: Session ID

        Returns:
            Anomaly detection result
        """
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {'error': 'Session does not exist'}

            anomalies = []
            risk_score = 0.0

            # Check session duration anomaly
            session_duration = datetime.now() - session.created_at
            if session_duration > timedelta(hours=12):
                anomalies.append('Session duration too long')
                risk_score += 0.3

            # Check activity frequency anomaly
            recent_activity = datetime.now() - session.last_activity
            if recent_activity > timedelta(hours=2):
                anomalies.append('Long period of inactivity')
                risk_score += 0.2

            # Check permission usage pattern
            user_actions = [
                attempt for attempt in self.access_log
                if attempt.user_id == session.user_id
                and attempt.timestamp > session.created_at
            ]

            if len(user_actions) > 100:  # High-frequency operations
                anomalies.append('Abnormal operation frequency')
                risk_score += 0.2

            # Check failed operation ratio
            failed_actions = [a for a in user_actions if not a.success]
            if user_actions and len(failed_actions) / len(user_actions) > 0.3:
                anomalies.append('Too high failed operation ratio')
                risk_score += 0.3

            return {
                'session_id': session_id,
                'anomalies': anomalies,
                'risk_score': min(risk_score, 1.0),
                'recommendation': self._get_anomaly_recommendation(risk_score)
            }

        except Exception as e:
            self.logger.error(f"Session anomaly detection failed: {e}")
            return {'error': str(e)}

    def _get_anomaly_recommendation(self, risk_score: float) -> str:
        """Get anomaly handling recommendation"""
        if risk_score >= 0.8:
            return "Recommend terminating session immediately"
        elif risk_score >= 0.5:
            return "Recommend enhanced monitoring"
        elif risk_score >= 0.3:
            return "Recommend monitoring session activity"
        else:
            return "Session is normal"

    def force_logout_user(self, user_id: str) -> int:
        """
        Force user to logout from all sessions

        Args:
            user_id: User ID

        Returns:
            Number of terminated sessions
        """
        try:
            terminated_sessions = []

            for session_id, session in self.active_sessions.items():
                if session.user_id == user_id:
                    terminated_sessions.append(session_id)

            for session_id in terminated_sessions:
                del self.active_sessions[session_id]

            if terminated_sessions:
                self._log_access_attempt(
                    user_id, "force_logout", "system",
                    True, None, f"Force logged out {len(terminated_sessions)} sessions"
                )

            self.logger.info(f"Force logged out user {user_id} from {len(terminated_sessions)} sessions")
            return len(terminated_sessions)

        except Exception as e:
            self.logger.error(f"Force logout failed: {e}")
            return 0

    def get_security_statistics(self) -> Dict[str, Any]:
        """
        Get security statistics

        Returns:
            Security statistics information
        """
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)

            # Count activity in the last 24 hours
            recent_attempts = [
                attempt for attempt in self.access_log
                if attempt.timestamp >= last_24h
            ]

            successful_logins = [
                attempt for attempt in recent_attempts
                if attempt.action in ['authenticate', 'authenticate_mfa'] and attempt.success
            ]

            failed_logins = [
                attempt for attempt in recent_attempts
                if attempt.action in ['authenticate', 'authenticate_mfa'] and not attempt.success
            ]

            # Count IP addresses
            unique_ips = set()
            for attempt in recent_attempts:
                if attempt.ip_address:
                    unique_ips.add(attempt.ip_address)

            # Count current sessions
            active_sessions_count = len(self.active_sessions)
            locked_users_count = len(self.locked_users)

            return {
                'current_status': {
                    'active_sessions': active_sessions_count,
                    'locked_users': locked_users_count,
                    'unique_ips_24h': len(unique_ips)
                },
                'last_24h_activity': {
                    'total_attempts': len(recent_attempts),
                    'successful_logins': len(successful_logins),
                    'failed_logins': len(failed_logins),
                    'success_rate': len(successful_logins) / len(recent_attempts) if recent_attempts else 0
                },
                'security_features': {
                    'mfa_enabled': hasattr(self, '_mfa_secrets') and bool(self._mfa_secrets),
                    'ip_whitelist_active': hasattr(self, '_ip_whitelist') and bool(self._ip_whitelist),
                    'ip_blacklist_active': hasattr(self, '_ip_blacklist') and bool(self._ip_blacklist),
                    'audit_logging': self.access_config['enable_audit_log']
                },
                'configuration': {
                    'session_timeout_minutes': self.access_config['session_timeout_minutes'],
                    'max_failed_attempts': self.access_config['max_failed_attempts'],
                    'lockout_duration_minutes': self.access_config['lockout_duration_minutes'],
                    'max_concurrent_sessions': self.access_config['max_concurrent_sessions']
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get security statistics: {e}")
            return {'error': str(e)}

    def export_audit_log(self, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Export audit log

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of audit logs
        """
        try:
            filtered_log = self.access_log

            if start_date:
                filtered_log = [
                    attempt for attempt in filtered_log
                    if attempt.timestamp >= start_date
                ]

            if end_date:
                filtered_log = [
                    attempt for attempt in filtered_log
                    if attempt.timestamp <= end_date
                ]

            return [
                {
                    'timestamp': attempt.timestamp.isoformat(),
                    'user_id': attempt.user_id,
                    'action': attempt.action,
                    'resource': attempt.resource,
                    'success': attempt.success,
                    'ip_address': attempt.ip_address,
                    'reason': attempt.reason
                }
                for attempt in filtered_log
            ]

        except Exception as e:
            self.logger.error(f"Failed to export audit log: {e}")
            return []

    def get_security_status(self) -> Dict[str, Any]:
        """Get security status"""
        return {
            'active_sessions': len(self.active_sessions),
            'locked_users': len(self.locked_users),
            'failed_attempts_count': sum(len(attempts) for attempts in self.failed_attempts.values()),
            'audit_log_entries': len(self.access_log),
            'authentication_required': self.access_config['require_authentication'],
            'session_timeout_minutes': self.access_config['session_timeout_minutes']
        }