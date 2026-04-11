// Mycelix Pulse iOS - Data Models

import Foundation

// MARK: - Email

struct Email: Identifiable, Codable {
    let id: UUID
    let senderName: String
    let senderEmail: String
    let recipientEmail: String
    let subject: String
    let body: String
    let preview: String
    let date: Date
    let trustScore: Double
    var isRead: Bool
    var isStarred: Bool
    let hasAttachments: Bool
    let attachments: [Attachment]
    let labels: [String]
    let threadId: UUID?
    let inReplyTo: UUID?
    let encryptionStatus: EncryptionStatus

    var formattedDate: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }

    var formattedFullDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .full
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Attachment

struct Attachment: Identifiable, Codable {
    let id: UUID
    let name: String
    let mimeType: String
    let size: Int64
    let url: URL?

    var icon: String {
        switch mimeType {
        case let type where type.starts(with: "image/"):
            return "photo"
        case let type where type.starts(with: "video/"):
            return "video"
        case let type where type.starts(with: "audio/"):
            return "music.note"
        case "application/pdf":
            return "doc.text"
        case let type where type.contains("spreadsheet") || type.contains("excel"):
            return "tablecells"
        case let type where type.contains("presentation") || type.contains("powerpoint"):
            return "rectangle.on.rectangle"
        case let type where type.contains("zip") || type.contains("archive"):
            return "archivebox"
        default:
            return "doc"
        }
    }

    var formattedSize: String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }
}

struct AttachmentItem: Identifiable {
    let id = UUID()
    let name: String
    let data: Data
    let mimeType: String

    var icon: String {
        switch mimeType {
        case let type where type.starts(with: "image/"):
            return "photo"
        default:
            return "doc"
        }
    }
}

// MARK: - Trust

struct TrustedContact: Identifiable, Codable {
    let id: UUID
    let name: String
    let email: String
    let trustScore: Double
    let verifiedAt: Date?
    let mutualConnections: Int
    let avatar: URL?
}

struct TrustRequest: Identifiable, Codable {
    let id: UUID
    let email: String
    let name: String?
    let message: String?
    let requestedAt: Date
}

struct ContactSuggestion: Identifiable, Codable {
    let id: UUID
    let email: String
    let name: String?
    let reason: String
    let mutualConnections: Int
}

// MARK: - Encryption

enum EncryptionStatus: String, Codable {
    case encrypted
    case signed
    case encryptedAndSigned
    case none

    var icon: String {
        switch self {
        case .encrypted:
            return "lock.fill"
        case .signed:
            return "checkmark.seal.fill"
        case .encryptedAndSigned:
            return "lock.shield.fill"
        case .none:
            return "lock.open"
        }
    }

    var description: String {
        switch self {
        case .encrypted:
            return "End-to-end encrypted"
        case .signed:
            return "Digitally signed"
        case .encryptedAndSigned:
            return "Encrypted and signed"
        case .none:
            return "Not encrypted"
        }
    }
}

// MARK: - Filter

enum MailFilter {
    case all
    case unread
    case starred
    case highTrust
    case attachments
    case label(String)
}

// MARK: - Email Template

struct EmailTemplate: Identifiable, Codable {
    let id: UUID
    let name: String
    let subject: String
    let body: String
    let variables: [String]
}

// MARK: - Folder

struct Folder: Identifiable, Codable {
    let id: UUID
    let name: String
    let icon: String
    let color: String?
    var unreadCount: Int
    let isSystem: Bool
}

// MARK: - User

struct User: Identifiable, Codable {
    let id: UUID
    let email: String
    let name: String
    let avatar: URL?
    let publicKey: String?
    let createdAt: Date
}

// MARK: - Auth

struct AuthTokens: Codable {
    let accessToken: String
    let refreshToken: String
    let expiresAt: Date
}

// MARK: - API Response

struct APIResponse<T: Codable>: Codable {
    let success: Bool
    let data: T?
    let error: APIError?
}

struct APIError: Codable, Error {
    let code: String
    let message: String
}

// MARK: - Notifications

struct PushNotificationPayload: Codable {
    let type: String
    let emailId: UUID?
    let title: String
    let body: String
    let trustScore: Double?
}
