#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data encryption manager
Provides data encryption, decryption, and key management functionality
"""

import os
import base64
import hashlib
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from ..utils.config import Config


class EncryptionManager:
    """Data encryption manager"""

    def __init__(self, config: Config):
        """
        Initialize encryption manager

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger('hibro.encryption_manager')

        # Encryption configuration
        self.encryption_config = {
            'algorithm': 'AES-256-GCM',
            'key_derivation': 'PBKDF2',
            'iterations': 100000,
            'salt_length': 32,
            'iv_length': 16,
            'tag_length': 16
        }

        # Key storage
        self._master_key: Optional[bytes] = None
        self._derived_keys: Dict[str, bytes] = {}
        self._key_file_path = self._get_key_file_path()

        # Initialize encryption system
        self._initialize_encryption()

    def _get_key_file_path(self) -> Path:
        """Get key file path"""
        data_dir = Path(self.config.data_directory)
        key_dir = data_dir / 'keys'
        key_dir.mkdir(parents=True, exist_ok=True)
        return key_dir / 'master.key'

    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            # Check if master key already exists
            if self._key_file_path.exists():
                self.logger.info("Found existing key file, auto-unlocking...")
                # Automatically unlock using default password
                self.unlock()
            else:
                self.logger.info("First run, will generate new master key")
                self._generate_master_key()
                # Master key is already set in _generate_master_key, system is unlocked

        except Exception as e:
            self.logger.error(f"Failed to initialize encryption system: {e}")
            raise

    def _generate_master_key(self) -> bytes:
        """Generate master key"""
        try:
            # Generate random master key
            master_key = os.urandom(32)  # 256-bit key

            # Encrypt master key using user password-derived key
            password = self._get_user_password()
            encrypted_master_key = self._encrypt_master_key(master_key, password)

            # Save encrypted master key
            with open(self._key_file_path, 'wb') as f:
                f.write(encrypted_master_key)

            # Set file permissions (owner read/write only)
            os.chmod(self._key_file_path, 0o600)

            self._master_key = master_key
            self.logger.info("Master key generation completed")

            return master_key

        except Exception as e:
            self.logger.error(f"Failed to generate master key: {e}")
            raise

    def _get_user_password(self) -> str:
        """Get user password"""
        # In production, this should obtain user password securely
        # For example, from environment variables, config file, or user input
        password = os.environ.get('MYJAVIS_PASSWORD')

        if not password:
            # Use system information to generate default password (not recommended for production)
            import platform
            import getpass

            system_info = f"{platform.node()}-{getpass.getuser()}-hibro"
            password = hashlib.sha256(system_info.encode()).hexdigest()[:32]

            self.logger.warning("Using default password, recommend setting MYJAVIS_PASSWORD environment variable")

        return password

    def _encrypt_master_key(self, master_key: bytes, password: str) -> bytes:
        """Encrypt master key using password"""
        # Generate salt
        salt = os.urandom(self.encryption_config['salt_length'])

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.encryption_config['iterations'],
            backend=default_backend()
        )
        key = kdf.derive(password.encode())

        # Encrypt master key using Fernet
        fernet = Fernet(base64.urlsafe_b64encode(key))
        encrypted_key = fernet.encrypt(master_key)

        # Return salt + encrypted master key
        return salt + encrypted_key

    def _decrypt_master_key(self, encrypted_data: bytes, password: str) -> bytes:
        """Decrypt master key using password"""
        # Extract salt
        salt = encrypted_data[:self.encryption_config['salt_length']]
        encrypted_key = encrypted_data[self.encryption_config['salt_length']:]

        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.encryption_config['iterations'],
            backend=default_backend()
        )
        key = kdf.derive(password.encode())

        # Decrypt master key using Fernet
        fernet = Fernet(base64.urlsafe_b64encode(key))
        master_key = fernet.decrypt(encrypted_key)

        return master_key

    def unlock(self, password: Optional[str] = None) -> bool:
        """
        Unlock encryption system

        Args:
            password: User password

        Returns:
            Whether unlock was successful
        """
        try:
            if not self._key_file_path.exists():
                self.logger.error("Key file does not exist")
                return False

            if password is None:
                password = self._get_user_password()

            # Read encrypted master key
            with open(self._key_file_path, 'rb') as f:
                encrypted_data = f.read()

            # Decrypt master key
            self._master_key = self._decrypt_master_key(encrypted_data, password)

            self.logger.info("Encryption system unlocked successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unlock encryption system: {e}")
            return False

    def is_unlocked(self) -> bool:
        """Check if encryption system is unlocked"""
        return self._master_key is not None

    def _derive_key(self, purpose: str) -> bytes:
        """
        Derive key for specific purpose

        Args:
            purpose: Key purpose

        Returns:
            Derived key
        """
        if not self.is_unlocked():
            raise RuntimeError("Encryption system not unlocked")

        if purpose in self._derived_keys:
            return self._derived_keys[purpose]

        # Derive key using HKDF
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=purpose.encode(),
            backend=default_backend()
        )

        derived_key = hkdf.derive(self._master_key)
        self._derived_keys[purpose] = derived_key

        return derived_key

    def encrypt_data(self, data: Union[str, bytes], purpose: str = 'general') -> bytes:
        """
        Encrypt data

        Args:
            data: Data to encrypt
            purpose: Encryption purpose

        Returns:
            Encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Get purpose-specific key
            key = self._derive_key(purpose)

            # Generate random IV
            iv = os.urandom(self.encryption_config['iv_length'])

            # Encrypt using AES-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            ciphertext = encryptor.update(data) + encryptor.finalize()

            # Return IV + authentication tag + ciphertext
            return iv + encryptor.tag + ciphertext

        except Exception as e:
            self.logger.error(f"Data encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: bytes, purpose: str = 'general') -> bytes:
        """
        Decrypt data

        Args:
            encrypted_data: Encrypted data
            purpose: Decryption purpose

        Returns:
            Decrypted data
        """
        try:
            # Get purpose-specific key
            key = self._derive_key(purpose)

            # Extract IV, tag, and ciphertext
            iv_length = self.encryption_config['iv_length']
            tag_length = self.encryption_config['tag_length']

            iv = encrypted_data[:iv_length]
            tag = encrypted_data[iv_length:iv_length + tag_length]
            ciphertext = encrypted_data[iv_length + tag_length:]

            # Decrypt using AES-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext

        except Exception as e:
            self.logger.error(f"Data decryption failed: {e}")
            raise

    def encrypt_string(self, text: str, purpose: str = 'general') -> str:
        """
        Encrypt string and return Base64 encoding

        Args:
            text: String to encrypt
            purpose: Encryption purpose

        Returns:
            Base64 encoded encrypted data
        """
        encrypted_data = self.encrypt_data(text, purpose)
        return base64.b64encode(encrypted_data).decode('ascii')

    def decrypt_string(self, encrypted_text: str, purpose: str = 'general') -> str:
        """
        Decrypt Base64 encoded string

        Args:
            encrypted_text: Base64 encoded encrypted data
            purpose: Decryption purpose

        Returns:
            Decrypted string
        """
        encrypted_data = base64.b64decode(encrypted_text.encode('ascii'))
        decrypted_data = self.decrypt_data(encrypted_data, purpose)
        return decrypted_data.decode('utf-8')

    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None,
                    purpose: str = 'file') -> Path:
        """
        Encrypt file

        Args:
            file_path: Source file path
            output_path: Output file path
            purpose: Encryption purpose

        Returns:
            Encrypted file path
        """
        try:
            if output_path is None:
                output_path = file_path.with_suffix(file_path.suffix + '.enc')

            # Read file content
            with open(file_path, 'rb') as f:
                data = f.read()

            # Encrypt data
            encrypted_data = self.encrypt_data(data, purpose)

            # Write encrypted file
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)

            # Set file permissions
            os.chmod(output_path, 0o600)

            self.logger.info(f"File encryption completed: {file_path} -> {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            raise

    def decrypt_file(self, encrypted_file_path: Path, output_path: Optional[Path] = None,
                    purpose: str = 'file') -> Path:
        """
        Decrypt file

        Args:
            encrypted_file_path: Encrypted file path
            output_path: Output file path
            purpose: Decryption purpose

        Returns:
            Decrypted file path
        """
        try:
            if output_path is None:
                # Remove .enc suffix
                if encrypted_file_path.suffix == '.enc':
                    output_path = encrypted_file_path.with_suffix('')
                else:
                    output_path = encrypted_file_path.with_suffix('.dec')

            # Read encrypted file
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()

            # Decrypt data
            decrypted_data = self.decrypt_data(encrypted_data, purpose)

            # Write decrypted file
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)

            self.logger.info(f"File decryption completed: {encrypted_file_path} -> {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            raise

    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change password

        Args:
            old_password: Old password
            new_password: New password

        Returns:
            Whether change was successful
        """
        try:
            # Verify old password
            if not self._key_file_path.exists():
                self.logger.error("Key file does not exist")
                return False

            # Decrypt master key using old password
            with open(self._key_file_path, 'rb') as f:
                encrypted_data = f.read()

            master_key = self._decrypt_master_key(encrypted_data, old_password)

            # Encrypt master key using new password
            new_encrypted_key = self._encrypt_master_key(master_key, new_password)

            # Save new encrypted key
            with open(self._key_file_path, 'wb') as f:
                f.write(new_encrypted_key)

            self.logger.info("Password change successful")
            return True

        except Exception as e:
            self.logger.error(f"Password change failed: {e}")
            return False

    def rotate_keys(self) -> bool:
        """
        Key rotation mechanism

        Returns:
            Whether rotation was successful
        """
        try:
            if not self.is_unlocked():
                raise RuntimeError("Encryption system not unlocked")

            # Generate new master key
            new_master_key = os.urandom(32)

            # Encrypt new master key using current password
            password = self._get_user_password()
            encrypted_new_key = self._encrypt_master_key(new_master_key, password)

            # Backup old key file
            backup_path = self._key_file_path.with_suffix('.key.backup')
            if self._key_file_path.exists():
                import shutil
                shutil.copy2(self._key_file_path, backup_path)

            # Save new key
            with open(self._key_file_path, 'wb') as f:
                f.write(encrypted_new_key)

            # Update master key in memory
            old_master_key = self._master_key
            self._master_key = new_master_key

            # Clear derived key cache, force regeneration
            self._derived_keys.clear()

            self.logger.info("Key rotation completed")
            return True

        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return False

    def detect_sensitive_data(self, text: str) -> Dict[str, Any]:
        """
        Detect sensitive data

        Args:
            text: Text to detect

        Returns:
            Detection result dictionary
        """
        import re

        sensitive_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            'ssn': r'\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'password_field': r'(?i)(password|passwd|pwd|secret|token|key)\s*[:=]\s*[^\s]+',
            'private_key': r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }

        detected = {}
        sensitivity_score = 0.0

        for pattern_name, pattern in sensitive_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected[pattern_name] = {
                    'count': len(matches),
                    'matches': matches[:3]  # Keep only first 3 matches for logging
                }

                # Calculate sensitivity score
                if pattern_name in ['credit_card', 'ssn', 'private_key']:
                    sensitivity_score += 1.0
                elif pattern_name in ['password_field', 'api_key']:
                    sensitivity_score += 0.8
                elif pattern_name in ['email', 'phone']:
                    sensitivity_score += 0.4
                else:
                    sensitivity_score += 0.2

        return {
            'has_sensitive_data': len(detected) > 0,
            'sensitivity_score': min(sensitivity_score, 1.0),
            'detected_types': detected,
            'recommendation': self._get_encryption_recommendation(sensitivity_score)
        }

    def _get_encryption_recommendation(self, sensitivity_score: float) -> str:
        """Get encryption recommendation"""
        if sensitivity_score >= 0.8:
            return "Strongly recommend encryption - High sensitivity data detected"
        elif sensitivity_score >= 0.4:
            return "Recommend encryption - Medium sensitivity data detected"
        elif sensitivity_score > 0:
            return "Consider encryption - Low sensitivity data detected"
        else:
            return "No encryption needed - No sensitive data detected"

    def auto_encrypt_if_sensitive(self, data: str, purpose: str = 'auto') -> Dict[str, Any]:
        """
        Automatically detect and encrypt sensitive data

        Args:
            data: Data to detect and possibly encrypt
            purpose: Encryption purpose

        Returns:
            Processing result dictionary
        """
        try:
            # Detect sensitive data
            detection_result = self.detect_sensitive_data(data)

            result = {
                'original_data': data,
                'detection_result': detection_result,
                'encrypted': False,
                'encrypted_data': None
            }

            # If sensitive data detected and sensitivity score is high, auto-encrypt
            if detection_result['sensitivity_score'] >= 0.4:
                encrypted_data = self.encrypt_string(data, purpose)
                result.update({
                    'encrypted': True,
                    'encrypted_data': encrypted_data,
                    'encryption_purpose': purpose
                })

                self.logger.info(f"Auto-encrypted sensitive data, sensitivity score: {detection_result['sensitivity_score']:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Auto-encryption detection failed: {e}")
            return {
                'original_data': data,
                'detection_result': {'has_sensitive_data': False, 'sensitivity_score': 0.0},
                'encrypted': False,
                'encrypted_data': None,
                'error': str(e)
            }

    def get_encryption_statistics(self) -> Dict[str, Any]:
        """
        Get encryption statistics

        Returns:
            Encryption statistics
        """
        try:
            stats = {
                'is_unlocked': self.is_unlocked(),
                'master_key_exists': self._key_file_path.exists(),
                'derived_keys_count': len(self._derived_keys),
                'encryption_algorithm': self.encryption_config['algorithm'],
                'key_derivation_method': self.encryption_config['key_derivation'],
                'pbkdf2_iterations': self.encryption_config['iterations']
            }

            if self._key_file_path.exists():
                import os
                stat_info = os.stat(self._key_file_path)
                stats.update({
                    'key_file_size': stat_info.st_size,
                    'key_file_permissions': oct(stat_info.st_mode)[-3:],
                    'key_file_modified': stat_info.st_mtime
                })

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get encryption statistics: {e}")
            return {'error': str(e)}

    def secure_delete_key(self) -> bool:
        """
        Securely delete key file

        Returns:
            Whether deletion was successful
        """
        try:
            if not self._key_file_path.exists():
                self.logger.warning("Key file does not exist")
                return True

            # Overwrite file content multiple times to ensure secure deletion
            file_size = self._key_file_path.stat().st_size

            with open(self._key_file_path, 'r+b') as f:
                # Overwrite with random data 3 times
                for _ in range(3):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())

                # Finally overwrite with zeros
                f.seek(0)
                f.write(b'\x00' * file_size)
                f.flush()
                os.fsync(f.fileno())

            # Delete file
            self._key_file_path.unlink()

            # Clear keys in memory
            self._master_key = None
            self._derived_keys.clear()

            self.logger.info("Key file securely deleted")
            return True

        except Exception as e:
            self.logger.error(f"Failed to securely delete key file: {e}")
            return False

    def export_public_info(self) -> Dict[str, Any]:
        """
        Export public information (excluding sensitive data)

        Returns:
            Public information dictionary
        """
        return {
            'encryption_algorithm': self.encryption_config['algorithm'],
            'key_derivation_method': self.encryption_config['key_derivation'],
            'pbkdf2_iterations': self.encryption_config['iterations'],
            'salt_length': self.encryption_config['salt_length'],
            'iv_length': self.encryption_config['iv_length'],
            'tag_length': self.encryption_config['tag_length'],
            'is_initialized': self._key_file_path.exists(),
            'is_unlocked': self.is_unlocked()
        }

    def get_encryption_info(self) -> Dict[str, Any]:
        """
        Get encryption information

        Returns:
            Encryption system information
        """
        return {
            'algorithm': self.encryption_config['algorithm'],
            'key_derivation': self.encryption_config['key_derivation'],
            'iterations': self.encryption_config['iterations'],
            'is_unlocked': self.is_unlocked(),
            'key_file_exists': self._key_file_path.exists(),
            'derived_keys_count': len(self._derived_keys)
        }

    def clear_keys(self):
        """Clear keys in memory"""
        self._master_key = None
        self._derived_keys.clear()
        self.logger.info("Memory keys cleared")