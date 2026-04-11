// Track CG: iOS Native App (SwiftUI)
// Mycelix Pulse - Decentralized communication with Web-of-Trust

import SwiftUI

@main
struct MycelixMailApp: App {
    @StateObject private var authService = AuthService()
    @StateObject private var mailService = MailService()
    @StateObject private var trustService = TrustService()
    @StateObject private var notificationService = NotificationService()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authService)
                .environmentObject(mailService)
                .environmentObject(trustService)
                .environmentObject(notificationService)
                .onAppear {
                    setupAppearance()
                    notificationService.requestPermission()
                }
        }
    }

    private func setupAppearance() {
        // Custom navigation bar appearance
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor.systemBackground
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
}

// MARK: - Content View

struct ContentView: View {
    @EnvironmentObject var authService: AuthService

    var body: some View {
        Group {
            if authService.isAuthenticated {
                MainTabView()
            } else {
                LoginView()
            }
        }
        .animation(.easeInOut, value: authService.isAuthenticated)
    }
}

// MARK: - Main Tab View

struct MainTabView: View {
    @State private var selectedTab = 0
    @EnvironmentObject var mailService: MailService

    var body: some View {
        TabView(selection: $selectedTab) {
            InboxView()
                .tabItem {
                    Label("Inbox", systemImage: "tray.fill")
                }
                .badge(mailService.unreadCount)
                .tag(0)

            TrustNetworkView()
                .tabItem {
                    Label("Trust", systemImage: "person.3.fill")
                }
                .tag(1)

            ComposeView()
                .tabItem {
                    Label("Compose", systemImage: "square.and.pencil")
                }
                .tag(2)

            SearchView()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }
                .tag(3)

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(4)
        }
        .accentColor(.mycelixPrimary)
    }
}

// MARK: - Login View

struct LoginView: View {
    @EnvironmentObject var authService: AuthService
    @State private var email = ""
    @State private var password = ""
    @State private var isLoading = false
    @State private var showError = false
    @State private var errorMessage = ""

    var body: some View {
        NavigationView {
            VStack(spacing: 32) {
                // Logo
                VStack(spacing: 8) {
                    Image(systemName: "envelope.badge.shield.half.filled")
                        .font(.system(size: 80))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.mycelixPrimary, .mycelixSecondary],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )

                    Text("Mycelix Pulse")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text("Decentralized Email with Trust")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 60)

                // Login Form
                VStack(spacing: 16) {
                    TextField("Email", text: $email)
                        .textFieldStyle(MycelixTextFieldStyle())
                        .keyboardType(.emailAddress)
                        .textContentType(.emailAddress)
                        .autocapitalization(.none)

                    SecureField("Password", text: $password)
                        .textFieldStyle(MycelixTextFieldStyle())
                        .textContentType(.password)

                    Button(action: login) {
                        HStack {
                            if isLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Text("Sign In")
                                    .fontWeight(.semibold)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.mycelixPrimary)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(isLoading || email.isEmpty || password.isEmpty)
                }
                .padding(.horizontal, 32)

                // Alternative login options
                VStack(spacing: 16) {
                    Text("Or continue with")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 16) {
                        SocialLoginButton(icon: "apple.logo", label: "Apple") {
                            // Apple Sign In
                        }

                        SocialLoginButton(icon: "key.fill", label: "Passkey") {
                            // Passkey authentication
                        }
                    }
                }
                .padding(.top, 16)

                Spacer()

                // Sign up link
                HStack {
                    Text("Don't have an account?")
                        .foregroundColor(.secondary)
                    NavigationLink("Sign Up") {
                        SignUpView()
                    }
                }
                .font(.subheadline)
                .padding(.bottom, 32)
            }
            .navigationBarHidden(true)
            .alert("Error", isPresented: $showError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
        }
    }

    private func login() {
        isLoading = true
        Task {
            do {
                try await authService.login(email: email, password: password)
            } catch {
                errorMessage = error.localizedDescription
                showError = true
            }
            isLoading = false
        }
    }
}

// MARK: - Inbox View

struct InboxView: View {
    @EnvironmentObject var mailService: MailService
    @State private var selectedEmail: Email?
    @State private var isRefreshing = false

    var body: some View {
        NavigationView {
            List {
                ForEach(mailService.emails) { email in
                    EmailRowView(email: email)
                        .onTapGesture {
                            selectedEmail = email
                        }
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) {
                                mailService.delete(email)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }

                            Button {
                                mailService.archive(email)
                            } label: {
                                Label("Archive", systemImage: "archivebox")
                            }
                            .tint(.orange)
                        }
                        .swipeActions(edge: .leading) {
                            Button {
                                mailService.toggleRead(email)
                            } label: {
                                Label(
                                    email.isRead ? "Unread" : "Read",
                                    systemImage: email.isRead ? "envelope.badge" : "envelope.open"
                                )
                            }
                            .tint(.blue)
                        }
                }
            }
            .listStyle(.plain)
            .refreshable {
                await mailService.refresh()
            }
            .navigationTitle("Inbox")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Menu {
                        Button("All Mail") { mailService.setFilter(.all) }
                        Button("Unread") { mailService.setFilter(.unread) }
                        Button("Starred") { mailService.setFilter(.starred) }
                        Button("High Trust") { mailService.setFilter(.highTrust) }
                    } label: {
                        Label("Filter", systemImage: "line.3.horizontal.decrease.circle")
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        // Edit mode
                    } label: {
                        Text("Edit")
                    }
                }
            }
            .sheet(item: $selectedEmail) { email in
                EmailDetailView(email: email)
            }
        }
    }
}

// MARK: - Email Row View

struct EmailRowView: View {
    let email: Email
    @EnvironmentObject var trustService: TrustService

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Trust indicator
            TrustBadge(score: email.trustScore)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(email.senderName)
                        .font(.headline)
                        .fontWeight(email.isRead ? .regular : .bold)

                    Spacer()

                    Text(email.formattedDate)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Text(email.subject)
                    .font(.subheadline)
                    .fontWeight(email.isRead ? .regular : .semibold)
                    .lineLimit(1)

                Text(email.preview)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            // Indicators
            VStack {
                if email.hasAttachments {
                    Image(systemName: "paperclip")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if email.isStarred {
                    Image(systemName: "star.fill")
                        .font(.caption)
                        .foregroundColor(.yellow)
                }
            }
        }
        .padding(.vertical, 8)
        .opacity(email.isRead ? 0.8 : 1.0)
    }
}

// MARK: - Trust Badge

struct TrustBadge: View {
    let score: Double

    var color: Color {
        switch score {
        case 0.8...1.0: return .green
        case 0.5..<0.8: return .orange
        default: return .red
        }
    }

    var body: some View {
        ZStack {
            Circle()
                .fill(color.opacity(0.2))
                .frame(width: 40, height: 40)

            Circle()
                .trim(from: 0, to: score)
                .stroke(color, lineWidth: 3)
                .frame(width: 36, height: 36)
                .rotationEffect(.degrees(-90))

            Text("\(Int(score * 100))")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(color)
        }
    }
}

// MARK: - Email Detail View

struct EmailDetailView: View {
    let email: Email
    @Environment(\.dismiss) var dismiss
    @EnvironmentObject var trustService: TrustService
    @State private var showTrustDetails = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Header
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            TrustBadge(score: email.trustScore)

                            VStack(alignment: .leading) {
                                Text(email.senderName)
                                    .font(.headline)
                                Text(email.senderEmail)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            Button {
                                showTrustDetails = true
                            } label: {
                                Image(systemName: "info.circle")
                            }
                        }

                        Text(email.subject)
                            .font(.title2)
                            .fontWeight(.bold)

                        Text(email.formattedFullDate)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Body
                    Text(email.body)
                        .font(.body)
                        .padding()

                    // Attachments
                    if !email.attachments.isEmpty {
                        AttachmentsView(attachments: email.attachments)
                    }
                }
                .padding()
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }

                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button {
                        // Reply
                    } label: {
                        Image(systemName: "arrowshape.turn.up.left")
                    }

                    Button {
                        // Forward
                    } label: {
                        Image(systemName: "arrowshape.turn.up.right")
                    }

                    Menu {
                        Button("Mark as Unread") { }
                        Button("Move to Folder") { }
                        Button("Add Label") { }
                        Divider()
                        Button("Report Spam", role: .destructive) { }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $showTrustDetails) {
                TrustDetailsView(email: email)
            }
        }
    }
}

// MARK: - Trust Network View

struct TrustNetworkView: View {
    @EnvironmentObject var trustService: TrustService
    @State private var searchText = ""

    var body: some View {
        NavigationView {
            List {
                Section("Your Trust Network") {
                    ForEach(trustService.trustedContacts) { contact in
                        TrustContactRow(contact: contact)
                    }
                }

                Section("Trust Requests") {
                    ForEach(trustService.pendingRequests) { request in
                        TrustRequestRow(request: request)
                    }
                }

                Section("Suggested Connections") {
                    ForEach(trustService.suggestions) { suggestion in
                        SuggestionRow(suggestion: suggestion)
                    }
                }
            }
            .searchable(text: $searchText, prompt: "Search contacts")
            .navigationTitle("Trust Network")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        // Add contact
                    } label: {
                        Image(systemName: "person.badge.plus")
                    }
                }
            }
        }
    }
}

// MARK: - Compose View

struct ComposeView: View {
    @State private var to = ""
    @State private var cc = ""
    @State private var subject = ""
    @State private var body = ""
    @State private var showCc = false
    @State private var attachments: [AttachmentItem] = []
    @State private var showAttachmentPicker = false
    @State private var selectedTemplate: EmailTemplate?
    @EnvironmentObject var mailService: MailService

    var body: some View {
        NavigationView {
            Form {
                Section {
                    HStack {
                        Text("To:")
                            .foregroundColor(.secondary)
                            .frame(width: 40, alignment: .leading)
                        TextField("Recipients", text: $to)
                            .textContentType(.emailAddress)
                            .keyboardType(.emailAddress)
                            .autocapitalization(.none)
                    }

                    if showCc {
                        HStack {
                            Text("Cc:")
                                .foregroundColor(.secondary)
                                .frame(width: 40, alignment: .leading)
                            TextField("CC Recipients", text: $cc)
                                .textContentType(.emailAddress)
                        }
                    }

                    HStack {
                        Text("Subject:")
                            .foregroundColor(.secondary)
                            .frame(width: 60, alignment: .leading)
                        TextField("Subject", text: $subject)
                    }
                }

                Section {
                    TextEditor(text: $body)
                        .frame(minHeight: 200)
                }

                if !attachments.isEmpty {
                    Section("Attachments") {
                        ForEach(attachments) { attachment in
                            HStack {
                                Image(systemName: attachment.icon)
                                Text(attachment.name)
                                Spacer()
                                Button {
                                    attachments.removeAll { $0.id == attachment.id }
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("New Message")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Menu {
                        Button("Save Draft") { saveDraft() }
                        Button("Discard", role: .destructive) { clearForm() }
                    } label: {
                        Text("Cancel")
                    }
                }

                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Button {
                        showAttachmentPicker = true
                    } label: {
                        Image(systemName: "paperclip")
                    }

                    Button {
                        showCc.toggle()
                    } label: {
                        Image(systemName: "chevron.down.circle")
                    }

                    Button {
                        sendEmail()
                    } label: {
                        Image(systemName: "paperplane.fill")
                    }
                    .disabled(to.isEmpty || subject.isEmpty)
                }
            }
            .sheet(isPresented: $showAttachmentPicker) {
                AttachmentPickerView(attachments: $attachments)
            }
        }
    }

    private func sendEmail() {
        Task {
            await mailService.send(
                to: to,
                cc: cc,
                subject: subject,
                body: body,
                attachments: attachments
            )
            clearForm()
        }
    }

    private func saveDraft() {
        // Save draft implementation
    }

    private func clearForm() {
        to = ""
        cc = ""
        subject = ""
        body = ""
        attachments = []
    }
}

// MARK: - Search View

struct SearchView: View {
    @State private var searchText = ""
    @State private var searchResults: [Email] = []
    @State private var isSearching = false
    @EnvironmentObject var mailService: MailService

    var body: some View {
        NavigationView {
            VStack {
                if searchText.isEmpty {
                    // Recent searches
                    VStack(spacing: 20) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)

                        Text("Search your mail")
                            .font(.headline)

                        Text("Find emails by sender, subject, or content")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                } else if isSearching {
                    ProgressView("Searching...")
                } else if searchResults.isEmpty {
                    Text("No results found")
                        .foregroundColor(.secondary)
                } else {
                    List(searchResults) { email in
                        EmailRowView(email: email)
                    }
                    .listStyle(.plain)
                }
            }
            .searchable(text: $searchText, prompt: "Search emails")
            .onChange(of: searchText) { newValue in
                performSearch(query: newValue)
            }
            .navigationTitle("Search")
        }
    }

    private func performSearch(query: String) {
        guard !query.isEmpty else {
            searchResults = []
            return
        }

        isSearching = true
        Task {
            searchResults = await mailService.search(query: query)
            isSearching = false
        }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @EnvironmentObject var authService: AuthService
    @AppStorage("notificationsEnabled") private var notificationsEnabled = true
    @AppStorage("darkModeEnabled") private var darkModeEnabled = false
    @AppStorage("biometricEnabled") private var biometricEnabled = false

    var body: some View {
        NavigationView {
            List {
                Section("Account") {
                    NavigationLink {
                        AccountSettingsView()
                    } label: {
                        Label("Account Settings", systemImage: "person.circle")
                    }

                    NavigationLink {
                        SecuritySettingsView()
                    } label: {
                        Label("Security & Privacy", systemImage: "lock.shield")
                    }

                    NavigationLink {
                        EncryptionSettingsView()
                    } label: {
                        Label("Encryption Keys", systemImage: "key")
                    }
                }

                Section("Preferences") {
                    Toggle(isOn: $notificationsEnabled) {
                        Label("Notifications", systemImage: "bell")
                    }

                    Toggle(isOn: $darkModeEnabled) {
                        Label("Dark Mode", systemImage: "moon")
                    }

                    Toggle(isOn: $biometricEnabled) {
                        Label("Face ID / Touch ID", systemImage: "faceid")
                    }

                    NavigationLink {
                        SignatureSettingsView()
                    } label: {
                        Label("Email Signature", systemImage: "signature")
                    }
                }

                Section("Trust Network") {
                    NavigationLink {
                        TrustSettingsView()
                    } label: {
                        Label("Trust Settings", systemImage: "person.3")
                    }

                    NavigationLink {
                        BlockedSendersView()
                    } label: {
                        Label("Blocked Senders", systemImage: "person.fill.xmark")
                    }
                }

                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }

                    NavigationLink {
                        PrivacyPolicyView()
                    } label: {
                        Text("Privacy Policy")
                    }

                    NavigationLink {
                        TermsOfServiceView()
                    } label: {
                        Text("Terms of Service")
                    }
                }

                Section {
                    Button("Sign Out", role: .destructive) {
                        authService.logout()
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

// MARK: - Supporting Views

struct SocialLoginButton: View {
    let icon: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                Text(label)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
        .foregroundColor(.primary)
    }
}

struct AttachmentsView: View {
    let attachments: [Attachment]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Attachments")
                .font(.headline)

            ForEach(attachments) { attachment in
                HStack {
                    Image(systemName: attachment.icon)
                        .foregroundColor(.blue)

                    VStack(alignment: .leading) {
                        Text(attachment.name)
                            .font(.subheadline)
                        Text(attachment.formattedSize)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Button {
                        // Download attachment
                    } label: {
                        Image(systemName: "arrow.down.circle")
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(8)
            }
        }
        .padding()
    }
}

// MARK: - Custom Styles

struct MycelixTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
    }
}

// MARK: - Color Extensions

extension Color {
    static let mycelixPrimary = Color(red: 0.29, green: 0.56, blue: 0.85)
    static let mycelixSecondary = Color(red: 0.58, green: 0.39, blue: 0.85)
}

// MARK: - Placeholder Views

struct SignUpView: View {
    var body: some View {
        Text("Sign Up")
    }
}

struct TrustDetailsView: View {
    let email: Email
    var body: some View {
        Text("Trust Details for \(email.senderName)")
    }
}

struct TrustContactRow: View {
    let contact: TrustedContact
    var body: some View {
        Text(contact.name)
    }
}

struct TrustRequestRow: View {
    let request: TrustRequest
    var body: some View {
        Text(request.email)
    }
}

struct SuggestionRow: View {
    let suggestion: ContactSuggestion
    var body: some View {
        Text(suggestion.email)
    }
}

struct AttachmentPickerView: View {
    @Binding var attachments: [AttachmentItem]
    var body: some View {
        Text("Attachment Picker")
    }
}

struct AccountSettingsView: View {
    var body: some View { Text("Account Settings") }
}

struct SecuritySettingsView: View {
    var body: some View { Text("Security Settings") }
}

struct EncryptionSettingsView: View {
    var body: some View { Text("Encryption Settings") }
}

struct SignatureSettingsView: View {
    var body: some View { Text("Signature Settings") }
}

struct TrustSettingsView: View {
    var body: some View { Text("Trust Settings") }
}

struct BlockedSendersView: View {
    var body: some View { Text("Blocked Senders") }
}

struct PrivacyPolicyView: View {
    var body: some View { Text("Privacy Policy") }
}

struct TermsOfServiceView: View {
    var body: some View { Text("Terms of Service") }
}
