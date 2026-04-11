// Mycelix Pulse Android - Data Models

package com.mycelix.mail.data.models

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.time.temporal.ChronoUnit

// MARK: - Email

data class Email(
    val id: String,
    val senderName: String,
    val senderEmail: String,
    val recipientEmail: String,
    val subject: String,
    val body: String,
    val preview: String,
    val date: LocalDateTime,
    val trustScore: Double,
    val isRead: Boolean,
    val isStarred: Boolean,
    val hasAttachments: Boolean,
    val attachments: List<Attachment>,
    val labels: List<String>,
    val threadId: String?,
    val inReplyTo: String?,
    val isEncrypted: Boolean
) {
    val formattedDate: String
        get() {
            val now = LocalDateTime.now()
            val daysDiff = ChronoUnit.DAYS.between(date, now)

            return when {
                daysDiff == 0L -> date.format(DateTimeFormatter.ofPattern("h:mm a"))
                daysDiff < 7 -> date.format(DateTimeFormatter.ofPattern("EEE"))
                date.year == now.year -> date.format(DateTimeFormatter.ofPattern("MMM d"))
                else -> date.format(DateTimeFormatter.ofPattern("MM/dd/yy"))
            }
        }

    val formattedFullDate: String
        get() = date.format(DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy 'at' h:mm a"))
}

// MARK: - Attachment

data class Attachment(
    val id: String,
    val name: String,
    val mimeType: String,
    val size: Long,
    val url: String?
) {
    val formattedSize: String
        get() {
            return when {
                size < 1024 -> "$size B"
                size < 1024 * 1024 -> "${size / 1024} KB"
                size < 1024 * 1024 * 1024 -> "${size / (1024 * 1024)} MB"
                else -> "${size / (1024 * 1024 * 1024)} GB"
            }
        }
}

// MARK: - Trust

data class TrustedContact(
    val id: String,
    val name: String,
    val email: String,
    val trustScore: Double,
    val verifiedAt: LocalDateTime?,
    val mutualConnections: Int,
    val avatar: String?,
    val isVerified: Boolean
)

data class TrustRequest(
    val id: String,
    val email: String,
    val name: String?,
    val message: String?,
    val requestedAt: LocalDateTime
)

data class ContactSuggestion(
    val id: String,
    val email: String,
    val name: String?,
    val reason: String,
    val mutualConnections: Int
)

// MARK: - User

data class User(
    val id: String,
    val email: String,
    val name: String,
    val avatar: String?,
    val publicKey: String?,
    val createdAt: LocalDateTime
)

// MARK: - Auth

data class AuthTokens(
    val accessToken: String,
    val refreshToken: String,
    val expiresAt: LocalDateTime
)

data class LoginRequest(
    val email: String,
    val password: String
)

data class LoginResponse(
    val user: User,
    val tokens: AuthTokens
)

// MARK: - API

data class ApiResponse<T>(
    val success: Boolean,
    val data: T?,
    val error: ApiError?
)

data class ApiError(
    val code: String,
    val message: String
)

// MARK: - Filter

enum class MailFilter {
    ALL,
    UNREAD,
    STARRED,
    HIGH_TRUST,
    ATTACHMENTS
}

// MARK: - Encryption

enum class EncryptionStatus {
    ENCRYPTED,
    SIGNED,
    ENCRYPTED_AND_SIGNED,
    NONE
}
