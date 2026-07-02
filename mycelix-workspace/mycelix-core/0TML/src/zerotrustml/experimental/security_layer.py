# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root"""
Security Layer for Hybrid ZeroTrustML

Implements:
- TLS for WebSocket connections
- Message signing and verification (Ed25519)
- Node authentication with JWT
- Certificate management
"""

import asyncio
import logging
import os
import ssl
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# Cryptography imports
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PrivateFormat, PublicFormat, NoEncryption
    )
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    import jwt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: cryptography not available. Install with: pip install cryptography PyJWT")


@dataclass
class NodeIdentity:
    """Identity information for a node"""
    node_id: int
    public_key: bytes
    certificate: Optional[bytes] = None

    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id,
            'public_key': self.public_key.hex(),
            'certificate': self.certificate.hex() if self.certificate else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NodeIdentity':
        return cls(
            node_id=data['node_id'],
            public_key=bytes.fromhex(data['public_key']),
            certificate=bytes.fromhex(data['certificate']) if data.get('certificate') else None
        )


class SecurityManager:
    """Manages cryptographic operations for secure communication"""

    def __init__(self, node_id: int, cert_dir: Optional[Path] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for security features")

        self.node_id = node_id
        self.cert_dir = cert_dir or Path(f"./certs/node_{node_id}")
        self.cert_dir.mkdir(parents=True, exist_ok=True)

        # Ed25519 signing keys
        self.signing_key: Optional[ed25519.Ed25519PrivateKey] = None
        self.verify_key: Optional[ed25519.Ed25519PublicKey] = None

        # TLS certificate and key
        self.tls_private_key: Optional[rsa.RSAPrivateKey] = None
        self.tls_certificate: Optional[x509.Certificate] = None

        # Trusted peer public keys
        self.trusted_peers: Dict[int, ed25519.Ed25519PublicKey] = {}

        # JWT secret for authentication tokens
        self.jwt_secret = self._load_or_generate_jwt_secret()

        # Initialize keys
        self._initialize_keys()

    def _load_or_generate_jwt_secret(self) -> bytes:
        """Load or generate JWT secret.

        Production deployments MUST set ZEROTRUSTML_JWT_SECRET (base64-encoded,
        >= 32 bytes decoded) rather than relying on file-based storage.
        """
        import base64
        import secrets as _secrets

        logger = logging.getLogger(__name__)

        # Prefer environment variable (production path)
        env_secret = os.environ.get("ZEROTRUSTML_JWT_SECRET", "").strip()
        if env_secret:
            decoded = base64.b64decode(env_secret)
            if len(decoded) < 32:
                raise ValueError(
                    "ZEROTRUSTML_JWT_SECRET must decode to >= 32 bytes"
                )
            return decoded

        # Fall back to file-based storage (dev/legacy)
        secret_file = self.cert_dir / "jwt_secret.bin"
        if secret_file.exists():
            logger.warning(
                "Loading JWT secret from disk (%s) — set "
                "ZEROTRUSTML_JWT_SECRET env var for production",
                secret_file,
            )
            return secret_file.read_bytes()

        # Generate new secret and persist with restrictive permissions
        secret = _secrets.token_bytes(32)
        secret_file.write_bytes(secret)
        try:
            os.chmod(secret_file, 0o600)
        except OSError:
            pass  # Best-effort on platforms that don't support chmod
        logger.warning(
            "Generated new JWT secret at %s (owner-only perms). "
            "Set ZEROTRUSTML_JWT_SECRET env var for production.",
            secret_file,
        )
        return secret

    def _initialize_keys(self):
        """Initialize or load cryptographic keys"""
        # Load or generate Ed25519 signing key
        signing_key_file = self.cert_dir / "signing_key.pem"
        verify_key_file = self.cert_dir / "verify_key.pem"

        if signing_key_file.exists():
            # Load existing key
            with open(signing_key_file, "rb") as f:
                self.signing_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            with open(verify_key_file, "rb") as f:
                self.verify_key = serialization.load_pem_public_key(f.read())
        else:
            # Generate new key
            self.signing_key = ed25519.Ed25519PrivateKey.generate()
            self.verify_key = self.signing_key.public_key()

            # Save keys
            with open(signing_key_file, "wb") as f:
                f.write(self.signing_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                ))
            with open(verify_key_file, "wb") as f:
                f.write(self.verify_key.public_bytes(
                    encoding=Encoding.PEM,
                    format=PublicFormat.SubjectPublicKeyInfo
                ))

        # Load or generate TLS certificate
        self._initialize_tls_certificate()

    def _initialize_tls_certificate(self):
        """Initialize or load TLS certificate for WebSocket connections"""
        cert_file = self.cert_dir / "tls_cert.pem"
        key_file = self.cert_dir / "tls_key.pem"

        if cert_file.exists() and key_file.exists():
            # Load existing certificate
            with open(cert_file, "rb") as f:
                self.tls_certificate = x509.load_pem_x509_certificate(f.read())
            with open(key_file, "rb") as f:
                self.tls_private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        else:
            # Generate self-signed certificate
            self.tls_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )

            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ZeroTrustML"),
                x509.NameAttribute(NameOID.COMMON_NAME, f"node_{self.node_id}"),
            ])

            self.tls_certificate = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(self.tls_private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.utcnow())
                .not_valid_after(datetime.utcnow() + timedelta(days=365))
                .sign(self.tls_private_key, hashes.SHA256())
            )

            # Save certificate and key
            with open(cert_file, "wb") as f:
                f.write(self.tls_certificate.public_bytes(Encoding.PEM))
            with open(key_file, "wb") as f:
                f.write(self.tls_private_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                ))

    def get_ssl_context(self, is_server: bool = True) -> ssl.SSLContext:
        """Create SSL context for WebSocket connections.

        Server mode: loads this node's cert/key pair; does not verify client
        hostname (standard for servers accepting many clients).

        Client mode: verifies the server's certificate and hostname by default.
        Set ZEROTRUSTML_INSECURE_TLS=1 to disable verification for LOCAL
        DEVELOPMENT ONLY.
        """
        logger = logging.getLogger(__name__)

        if is_server:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(
                self.cert_dir / "tls_cert.pem",
                self.cert_dir / "tls_key.pem"
            )
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

            insecure = os.environ.get("ZEROTRUSTML_INSECURE_TLS", "").strip()
            if insecure == "1":
                logger.warning(
                    "*** ZEROTRUSTML_INSECURE_TLS is set — TLS certificate "
                    "verification DISABLED. DO NOT use this in production! ***"
                )
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            else:
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED

        return context

    def sign_message(self, message: bytes) -> bytes:
        """Sign a message with Ed25519 private key"""
        if not self.signing_key:
            raise RuntimeError("Signing key not initialized")

        return self.signing_key.sign(message)

    def verify_message(self, message: bytes, signature: bytes, peer_id: int) -> bool:
        """Verify message signature from a peer"""
        peer_key = self.trusted_peers.get(peer_id)
        if not peer_key:
            return False

        try:
            peer_key.verify(signature, message)
            return True
        except Exception:
            return False

    def add_trusted_peer(self, peer_id: int, public_key: bytes):
        """Add a trusted peer's public key"""
        try:
            key = serialization.load_pem_public_key(public_key)
            if isinstance(key, ed25519.Ed25519PublicKey):
                self.trusted_peers[peer_id] = key
        except Exception as e:
            print(f"Error adding trusted peer {peer_id}: {e}")

    def generate_auth_token(self, peer_id: int, expires_in: int = 3600) -> str:
        """Generate JWT authentication token for a peer"""
        payload = {
            'node_id': self.node_id,
            'peer_id': peer_id,
            'iat': int(time.time()),
            'exp': int(time.time()) + expires_in
        }

        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def verify_auth_token(self, token: str) -> Optional[Dict]:
        """Verify JWT authentication token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            print("Auth token expired")
            return None
        except jwt.InvalidTokenError:
            print("Invalid auth token")
            return None

    def get_identity(self) -> NodeIdentity:
        """Get this node's identity"""
        return NodeIdentity(
            node_id=self.node_id,
            public_key=self.verify_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            ),
            certificate=self.tls_certificate.public_bytes(Encoding.PEM)
        )

    def export_public_identity(self) -> dict:
        """Export public identity for sharing with peers"""
        return self.get_identity().to_dict()


class SecureMessageWrapper:
    """Wraps messages with cryptographic signatures"""

    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager

    def wrap_message(self, message_data: dict) -> dict:
        """Wrap message with signature and sender info"""
        # Serialize message
        message_bytes = json.dumps(message_data, sort_keys=True).encode()

        # Sign message
        signature = self.security.sign_message(message_bytes)

        # Create wrapped message
        wrapped = {
            'sender_id': self.security.node_id,
            'message': message_data,
            'signature': signature.hex(),
            'timestamp': time.time()
        }

        return wrapped

    def unwrap_message(self, wrapped_message: dict) -> Optional[dict]:
        """Unwrap and verify signed message"""
        try:
            sender_id = wrapped_message['sender_id']
            message = wrapped_message['message']
            signature = bytes.fromhex(wrapped_message['signature'])

            # Serialize message the same way
            message_bytes = json.dumps(message, sort_keys=True).encode()

            # Verify signature
            if not self.security.verify_message(message_bytes, signature, sender_id):
                print(f"Signature verification failed for message from {sender_id}")
                return None

            # Check timestamp (prevent replay attacks).
            # Default 60s — NIST SP 800-63B recommends short replay windows
            # for authentication protocols to limit the utility of captured tokens.
            # Override via ZEROTRUSTML_REPLAY_WINDOW_SECS env var if needed.
            replay_window = int(os.environ.get("ZEROTRUSTML_REPLAY_WINDOW_SECS", "60"))
            msg_time = wrapped_message['timestamp']
            if abs(time.time() - msg_time) > replay_window:
                print(f"Message from {sender_id} too old or from future")
                return None

            return message

        except Exception as e:
            print(f"Error unwrapping message: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Create security manager for node 1
    security1 = SecurityManager(node_id=1)
    security2 = SecurityManager(node_id=2)

    print("Node 1 Identity:")
    identity1 = security1.get_identity()
    print(f"  Node ID: {identity1.node_id}")
    print(f"  Public Key: {identity1.public_key.hex()[:32]}...")

    # Add node 2 as trusted peer
    security1.add_trusted_peer(2, identity2.public_key)
    security2.add_trusted_peer(1, identity1.public_key)

    # Test message signing
    wrapper1 = SecureMessageWrapper(security1)
    wrapper2 = SecureMessageWrapper(security2)

    original_message = {'type': 'gradient', 'data': [1, 2, 3]}
    wrapped = wrapper1.wrap_message(original_message)
    print(f"\nSigned message: {json.dumps(wrapped, indent=2)[:200]}...")

    # Verify message
    verified = wrapper2.unwrap_message(wrapped)
    if verified:
        print(f"\n✓ Message verified successfully")
        print(f"  Verified data: {verified}")
    else:
        print(f"\n✗ Message verification failed")

    # Test authentication tokens
    token = security1.generate_auth_token(peer_id=2)
    print(f"\nAuth token: {token[:50]}...")

    verified_payload = security1.verify_auth_token(token)
    if verified_payload:
        print(f"✓ Token verified: {verified_payload}")

    # TLS context
    ssl_context = security1.get_ssl_context(is_server=True)
    print(f"\n✓ SSL context created for secure WebSocket connections")

    print("\n" + "="*60)
    print("Security Layer initialized successfully")
    print("="*60)