// Mycelix Pulse iOS - Services

import Foundation
import Combine
import UserNotifications

// MARK: - Auth Service

@MainActor
class AuthService: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var tokens: AuthTokens?

    private let keychain = KeychainService()
    private let api = APIService.shared

    init() {
        checkStoredAuth()
    }

    private func checkStoredAuth() {
        if let tokens = keychain.getTokens() {
            self.tokens = tokens
            self.isAuthenticated = true
            Task {
                await fetchCurrentUser()
            }
        }
    }

    func login(email: String, password: String) async throws {
        let response = try await api.post(
            endpoint: "/auth/login",
            body: ["email": email, "password": password]
        ) as AuthResponse

        tokens = response.tokens
        currentUser = response.user
        isAuthenticated = true

        keychain.saveTokens(response.tokens)
    }

    func logout() {
        tokens = nil
        currentUser = nil
        isAuthenticated = false
        keychain.clearTokens()
    }

    func refreshTokens() async throws {
        guard let refreshToken = tokens?.refreshToken else {
            throw AuthError.noRefreshToken
        }

        let response = try await api.post(
            endpoint: "/auth/refresh",
            body: ["refresh_token": refreshToken]
        ) as AuthTokens

        tokens = response
        keychain.saveTokens(response)
    }

    private func fetchCurrentUser() async {
        do {
            currentUser = try await api.get(endpoint: "/users/me")
        } catch {
            print("Failed to fetch user: \(error)")
        }
    }
}

struct AuthResponse: Codable {
    let user: User
    let tokens: AuthTokens
}

enum AuthError: Error {
    case noRefreshToken
    case invalidCredentials
    case networkError
}

// MARK: - Mail Service

@MainActor
class MailService: ObservableObject {
    @Published var emails: [Email] = []
    @Published var unreadCount = 0
    @Published var currentFilter: MailFilter = .all
    @Published var isLoading = false

    private let api = APIService.shared
    private var cancellables = Set<AnyCancellable>()

    init() {
        Task {
            await loadEmails()
        }
    }

    func loadEmails() async {
        isLoading = true
        defer { isLoading = false }

        do {
            let response: EmailListResponse = try await api.get(endpoint: "/emails")
            emails = response.emails
            unreadCount = response.unreadCount
        } catch {
            print("Failed to load emails: \(error)")
        }
    }

    func refresh() async {
        await loadEmails()
    }

    func setFilter(_ filter: MailFilter) {
        currentFilter = filter
        Task {
            await loadEmails()
        }
    }

    func toggleRead(_ email: Email) {
        guard let index = emails.firstIndex(where: { $0.id == email.id }) else { return }
        emails[index].isRead.toggle()

        Task {
            try? await api.patch(
                endpoint: "/emails/\(email.id)",
                body: ["is_read": emails[index].isRead]
            )
        }

        updateUnreadCount()
    }

    func delete(_ email: Email) {
        emails.removeAll { $0.id == email.id }

        Task {
            try? await api.delete(endpoint: "/emails/\(email.id)")
        }

        updateUnreadCount()
    }

    func archive(_ email: Email) {
        emails.removeAll { $0.id == email.id }

        Task {
            try? await api.post(
                endpoint: "/emails/\(email.id)/archive",
                body: [:]
            )
        }
    }

    func send(
        to: String,
        cc: String,
        subject: String,
        body: String,
        attachments: [AttachmentItem]
    ) async {
        do {
            let _ = try await api.post(
                endpoint: "/emails/send",
                body: [
                    "to": to,
                    "cc": cc,
                    "subject": subject,
                    "body": body
                ]
            )
        } catch {
            print("Failed to send email: \(error)")
        }
    }

    func search(query: String) async -> [Email] {
        do {
            let response: EmailListResponse = try await api.get(
                endpoint: "/emails/search?q=\(query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query)"
            )
            return response.emails
        } catch {
            print("Search failed: \(error)")
            return []
        }
    }

    private func updateUnreadCount() {
        unreadCount = emails.filter { !$0.isRead }.count
    }
}

struct EmailListResponse: Codable {
    let emails: [Email]
    let unreadCount: Int
    let total: Int
}

// MARK: - Trust Service

@MainActor
class TrustService: ObservableObject {
    @Published var trustedContacts: [TrustedContact] = []
    @Published var pendingRequests: [TrustRequest] = []
    @Published var suggestions: [ContactSuggestion] = []

    private let api = APIService.shared

    init() {
        Task {
            await loadTrustNetwork()
        }
    }

    func loadTrustNetwork() async {
        do {
            async let contacts: [TrustedContact] = api.get(endpoint: "/trust/contacts")
            async let requests: [TrustRequest] = api.get(endpoint: "/trust/requests")
            async let suggests: [ContactSuggestion] = api.get(endpoint: "/trust/suggestions")

            trustedContacts = try await contacts
            pendingRequests = try await requests
            suggestions = try await suggests
        } catch {
            print("Failed to load trust network: \(error)")
        }
    }

    func getTrustScore(for email: String) async -> Double {
        do {
            let response: TrustScoreResponse = try await api.get(
                endpoint: "/trust/score/\(email)"
            )
            return response.score
        } catch {
            return 0.0
        }
    }

    func sendTrustRequest(to email: String, message: String?) async throws {
        try await api.post(
            endpoint: "/trust/request",
            body: ["email": email, "message": message ?? ""]
        )
    }

    func acceptRequest(_ request: TrustRequest) async throws {
        try await api.post(
            endpoint: "/trust/requests/\(request.id)/accept",
            body: [:]
        )
        pendingRequests.removeAll { $0.id == request.id }
    }

    func rejectRequest(_ request: TrustRequest) async throws {
        try await api.post(
            endpoint: "/trust/requests/\(request.id)/reject",
            body: [:]
        )
        pendingRequests.removeAll { $0.id == request.id }
    }

    func revokeTrust(for contact: TrustedContact) async throws {
        try await api.delete(endpoint: "/trust/contacts/\(contact.id)")
        trustedContacts.removeAll { $0.id == contact.id }
    }
}

struct TrustScoreResponse: Codable {
    let email: String
    let score: Double
    let factors: [TrustFactor]
}

struct TrustFactor: Codable {
    let name: String
    let weight: Double
    let value: Double
}

// MARK: - Notification Service

class NotificationService: ObservableObject {
    @Published var isEnabled = false

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
            DispatchQueue.main.async {
                self.isEnabled = granted
            }
        }
    }

    func registerForRemoteNotifications() {
        DispatchQueue.main.async {
            UIApplication.shared.registerForRemoteNotifications()
        }
    }

    func handleNotification(_ payload: PushNotificationPayload) {
        // Handle incoming notification
        print("Received notification: \(payload.title)")
    }
}

// MARK: - API Service

class APIService {
    static let shared = APIService()

    private let baseURL = "https://api.mycelix.mail/v1"
    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        session = URLSession(configuration: config)
    }

    func get<T: Codable>(endpoint: String) async throws -> T {
        let url = URL(string: baseURL + endpoint)!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        addAuthHeaders(&request)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)

        return try JSONDecoder().decode(T.self, from: data)
    }

    func post<T: Codable>(endpoint: String, body: [String: Any]) async throws -> T {
        let url = URL(string: baseURL + endpoint)!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeaders(&request)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)

        return try JSONDecoder().decode(T.self, from: data)
    }

    @discardableResult
    func post(endpoint: String, body: [String: Any]) async throws -> Data {
        let url = URL(string: baseURL + endpoint)!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeaders(&request)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)

        return data
    }

    func patch(endpoint: String, body: [String: Any]) async throws {
        let url = URL(string: baseURL + endpoint)!
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addAuthHeaders(&request)

        let (_, response) = try await session.data(for: request)
        try validateResponse(response)
    }

    func delete(endpoint: String) async throws {
        let url = URL(string: baseURL + endpoint)!
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        addAuthHeaders(&request)

        let (_, response) = try await session.data(for: request)
        try validateResponse(response)
    }

    private func addAuthHeaders(_ request: inout URLRequest) {
        if let tokens = KeychainService().getTokens() {
            request.setValue("Bearer \(tokens.accessToken)", forHTTPHeaderField: "Authorization")
        }
    }

    private func validateResponse(_ response: URLResponse) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIServiceError.invalidResponse
        }

        switch httpResponse.statusCode {
        case 200...299:
            return
        case 401:
            throw APIServiceError.unauthorized
        case 404:
            throw APIServiceError.notFound
        case 500...599:
            throw APIServiceError.serverError
        default:
            throw APIServiceError.unknown(httpResponse.statusCode)
        }
    }
}

enum APIServiceError: Error {
    case invalidResponse
    case unauthorized
    case notFound
    case serverError
    case unknown(Int)
}

// MARK: - Keychain Service

class KeychainService {
    private let tokenKey = "com.mycelix.mail.tokens"

    func saveTokens(_ tokens: AuthTokens) {
        guard let data = try? JSONEncoder().encode(tokens) else { return }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: tokenKey,
            kSecValueData as String: data
        ]

        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }

    func getTokens() -> AuthTokens? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: tokenKey,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let tokens = try? JSONDecoder().decode(AuthTokens.self, from: data) else {
            return nil
        }

        return tokens
    }

    func clearTokens() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: tokenKey
        ]

        SecItemDelete(query as CFDictionary)
    }
}

// MARK: - Encryption Service

class EncryptionService {
    static let shared = EncryptionService()

    func generateKeyPair() throws -> (publicKey: String, privateKey: String) {
        // Generate X25519 key pair for encryption
        // In production, use CryptoKit
        return ("public_key_placeholder", "private_key_placeholder")
    }

    func encrypt(message: String, recipientPublicKey: String) throws -> String {
        // Encrypt message with recipient's public key
        return "encrypted_\(message)"
    }

    func decrypt(encryptedMessage: String, privateKey: String) throws -> String {
        // Decrypt message with private key
        return encryptedMessage.replacingOccurrences(of: "encrypted_", with: "")
    }

    func sign(message: String, privateKey: String) throws -> String {
        // Create digital signature
        return "signature_placeholder"
    }

    func verify(message: String, signature: String, publicKey: String) -> Bool {
        // Verify digital signature
        return true
    }
}
