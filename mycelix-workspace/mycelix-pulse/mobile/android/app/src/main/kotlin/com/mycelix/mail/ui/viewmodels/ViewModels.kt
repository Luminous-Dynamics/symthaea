// Mycelix Pulse Android - ViewModels

package com.mycelix.mail.ui.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.mycelix.mail.data.models.*
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

// MARK: - Auth ViewModel

@HiltViewModel
class AuthViewModel @Inject constructor() : ViewModel() {

    private val _isAuthenticated = MutableStateFlow(false)
    val isAuthenticated: StateFlow<Boolean> = _isAuthenticated.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    private val _currentUser = MutableStateFlow<User?>(null)
    val currentUser: StateFlow<User?> = _currentUser.asStateFlow()

    fun login(email: String, password: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null

            try {
                // API call would go here
                // val response = authRepository.login(email, password)
                // _currentUser.value = response.user
                // tokenManager.saveTokens(response.tokens)
                _isAuthenticated.value = true
            } catch (e: Exception) {
                _error.value = e.message ?: "Login failed"
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun logout() {
        viewModelScope.launch {
            // tokenManager.clearTokens()
            _currentUser.value = null
            _isAuthenticated.value = false
        }
    }

    fun checkAuth() {
        viewModelScope.launch {
            // Check stored tokens
            // if (tokenManager.hasValidTokens()) {
            //     _isAuthenticated.value = true
            //     fetchCurrentUser()
            // }
        }
    }
}

// MARK: - Mail ViewModel

@HiltViewModel
class MailViewModel @Inject constructor() : ViewModel() {

    private val _emails = MutableStateFlow<List<Email>>(emptyList())
    val emails: StateFlow<List<Email>> = _emails.asStateFlow()

    private val _selectedEmail = MutableStateFlow<Email?>(null)
    val selectedEmail: StateFlow<Email?> = _selectedEmail.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _unreadCount = MutableStateFlow(0)
    val unreadCount: StateFlow<Int> = _unreadCount.asStateFlow()

    private val _currentFilter = MutableStateFlow(MailFilter.ALL)
    val currentFilter: StateFlow<MailFilter> = _currentFilter.asStateFlow()

    init {
        loadEmails()
    }

    fun loadEmails() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                // val response = mailRepository.getEmails(filter = _currentFilter.value)
                // _emails.value = response.emails
                // _unreadCount.value = response.unreadCount

                // Mock data for demonstration
                _emails.value = createMockEmails()
                _unreadCount.value = _emails.value.count { !it.isRead }
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun loadEmail(emailId: String) {
        viewModelScope.launch {
            try {
                // val email = mailRepository.getEmail(emailId)
                // _selectedEmail.value = email
                _selectedEmail.value = _emails.value.find { it.id == emailId }
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun refresh() {
        loadEmails()
    }

    fun setFilter(filter: MailFilter) {
        _currentFilter.value = filter
        loadEmails()
    }

    fun deleteEmail(emailId: String) {
        viewModelScope.launch {
            try {
                // mailRepository.deleteEmail(emailId)
                _emails.value = _emails.value.filter { it.id != emailId }
                updateUnreadCount()
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun archiveEmail(emailId: String) {
        viewModelScope.launch {
            try {
                // mailRepository.archiveEmail(emailId)
                _emails.value = _emails.value.filter { it.id != emailId }
                updateUnreadCount()
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun toggleRead(emailId: String) {
        viewModelScope.launch {
            val email = _emails.value.find { it.id == emailId } ?: return@launch
            val updated = email.copy(isRead = !email.isRead)
            // mailRepository.updateEmail(updated)
            _emails.value = _emails.value.map { if (it.id == emailId) updated else it }
            updateUnreadCount()
        }
    }

    fun toggleStar(emailId: String) {
        viewModelScope.launch {
            val email = _emails.value.find { it.id == emailId } ?: return@launch
            val updated = email.copy(isStarred = !email.isStarred)
            // mailRepository.updateEmail(updated)
            _emails.value = _emails.value.map { if (it.id == emailId) updated else it }
        }
    }

    private fun updateUnreadCount() {
        _unreadCount.value = _emails.value.count { !it.isRead }
    }

    private fun createMockEmails(): List<Email> {
        return listOf(
            Email(
                id = "1",
                senderName = "Alice Smith",
                senderEmail = "alice@example.com",
                recipientEmail = "me@mycelix.mail",
                subject = "Project Update - Q4 Planning",
                body = "Hi,\n\nI wanted to share the latest updates on our Q4 planning...",
                preview = "I wanted to share the latest updates on our Q4 planning and discuss the timeline for the upcoming...",
                date = java.time.LocalDateTime.now().minusHours(2),
                trustScore = 0.92,
                isRead = false,
                isStarred = true,
                hasAttachments = true,
                attachments = listOf(
                    Attachment("a1", "Q4_Plan.pdf", "application/pdf", 2048576, null)
                ),
                labels = listOf("work", "important"),
                threadId = null,
                inReplyTo = null,
                isEncrypted = true
            ),
            Email(
                id = "2",
                senderName = "Bob Johnson",
                senderEmail = "bob@company.com",
                recipientEmail = "me@mycelix.mail",
                subject = "Re: Meeting Tomorrow",
                body = "Sounds good! I'll see you at 3pm.",
                preview = "Sounds good! I'll see you at 3pm. Let me know if anything changes.",
                date = java.time.LocalDateTime.now().minusDays(1),
                trustScore = 0.78,
                isRead = true,
                isStarred = false,
                hasAttachments = false,
                attachments = emptyList(),
                labels = listOf("work"),
                threadId = "thread-123",
                inReplyTo = "prev-email-id",
                isEncrypted = false
            ),
            Email(
                id = "3",
                senderName = "Newsletter",
                senderEmail = "newsletter@techdigest.com",
                recipientEmail = "me@mycelix.mail",
                subject = "Weekly Tech Digest",
                body = "This week in tech...",
                preview = "This week in tech: AI breakthroughs, new smartphone releases, and more...",
                date = java.time.LocalDateTime.now().minusDays(3),
                trustScore = 0.45,
                isRead = true,
                isStarred = false,
                hasAttachments = false,
                attachments = emptyList(),
                labels = listOf("newsletter"),
                threadId = null,
                inReplyTo = null,
                isEncrypted = false
            )
        )
    }
}

// MARK: - Trust ViewModel

@HiltViewModel
class TrustViewModel @Inject constructor() : ViewModel() {

    private val _trustedContacts = MutableStateFlow<List<TrustedContact>>(emptyList())
    val trustedContacts: StateFlow<List<TrustedContact>> = _trustedContacts.asStateFlow()

    private val _pendingRequests = MutableStateFlow<List<TrustRequest>>(emptyList())
    val pendingRequests: StateFlow<List<TrustRequest>> = _pendingRequests.asStateFlow()

    private val _suggestions = MutableStateFlow<List<ContactSuggestion>>(emptyList())
    val suggestions: StateFlow<List<ContactSuggestion>> = _suggestions.asStateFlow()

    init {
        loadTrustNetwork()
    }

    fun loadTrustNetwork() {
        viewModelScope.launch {
            try {
                // val contacts = trustRepository.getTrustedContacts()
                // val requests = trustRepository.getPendingRequests()
                // val suggestions = trustRepository.getSuggestions()

                // Mock data
                _trustedContacts.value = listOf(
                    TrustedContact(
                        id = "1",
                        name = "Alice Smith",
                        email = "alice@example.com",
                        trustScore = 0.92,
                        verifiedAt = java.time.LocalDateTime.now().minusDays(30),
                        mutualConnections = 5,
                        avatar = null,
                        isVerified = true
                    ),
                    TrustedContact(
                        id = "2",
                        name = "Bob Johnson",
                        email = "bob@company.com",
                        trustScore = 0.78,
                        verifiedAt = null,
                        mutualConnections = 2,
                        avatar = null,
                        isVerified = false
                    )
                )

                _pendingRequests.value = listOf(
                    TrustRequest(
                        id = "r1",
                        email = "carol@newcontact.com",
                        name = "Carol Williams",
                        message = "Hi! We met at the conference last week.",
                        requestedAt = java.time.LocalDateTime.now().minusHours(4)
                    )
                )
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun acceptRequest(requestId: String) {
        viewModelScope.launch {
            try {
                // trustRepository.acceptRequest(requestId)
                _pendingRequests.value = _pendingRequests.value.filter { it.id != requestId }
                loadTrustNetwork()
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun rejectRequest(requestId: String) {
        viewModelScope.launch {
            try {
                // trustRepository.rejectRequest(requestId)
                _pendingRequests.value = _pendingRequests.value.filter { it.id != requestId }
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun sendTrustRequest(email: String, message: String?) {
        viewModelScope.launch {
            try {
                // trustRepository.sendRequest(email, message)
            } catch (e: Exception) {
                // Handle error
            }
        }
    }

    fun revokeTrust(contactId: String) {
        viewModelScope.launch {
            try {
                // trustRepository.revokeTrust(contactId)
                _trustedContacts.value = _trustedContacts.value.filter { it.id != contactId }
            } catch (e: Exception) {
                // Handle error
            }
        }
    }
}

// MARK: - Compose ViewModel

@HiltViewModel
class ComposeViewModel @Inject constructor() : ViewModel() {

    private val _isSending = MutableStateFlow(false)
    val isSending: StateFlow<Boolean> = _isSending.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    fun send(to: String, cc: String, subject: String, body: String) {
        viewModelScope.launch {
            _isSending.value = true
            _error.value = null

            try {
                // mailRepository.sendEmail(
                //     SendEmailRequest(to, cc, subject, body)
                // )
            } catch (e: Exception) {
                _error.value = e.message ?: "Failed to send email"
            } finally {
                _isSending.value = false
            }
        }
    }

    fun saveDraft(to: String, cc: String, subject: String, body: String) {
        viewModelScope.launch {
            try {
                // mailRepository.saveDraft(...)
            } catch (e: Exception) {
                // Handle error
            }
        }
    }
}

// MARK: - Search ViewModel

@HiltViewModel
class SearchViewModel @Inject constructor() : ViewModel() {

    private val _results = MutableStateFlow<List<Email>>(emptyList())
    val results: StateFlow<List<Email>> = _results.asStateFlow()

    private val _isSearching = MutableStateFlow(false)
    val isSearching: StateFlow<Boolean> = _isSearching.asStateFlow()

    private val _recentSearches = MutableStateFlow<List<String>>(emptyList())
    val recentSearches: StateFlow<List<String>> = _recentSearches.asStateFlow()

    fun search(query: String) {
        if (query.isBlank()) {
            _results.value = emptyList()
            return
        }

        viewModelScope.launch {
            _isSearching.value = true
            try {
                // val results = mailRepository.search(query)
                // _results.value = results
                // saveRecentSearch(query)
                _results.value = emptyList() // Mock empty for now
            } catch (e: Exception) {
                // Handle error
            } finally {
                _isSearching.value = false
            }
        }
    }

    fun clearRecentSearches() {
        _recentSearches.value = emptyList()
    }
}

// MARK: - Settings ViewModel

@HiltViewModel
class SettingsViewModel @Inject constructor() : ViewModel() {

    private val _isDarkMode = MutableStateFlow(false)
    val isDarkMode: StateFlow<Boolean> = _isDarkMode.asStateFlow()

    private val _notificationsEnabled = MutableStateFlow(true)
    val notificationsEnabled: StateFlow<Boolean> = _notificationsEnabled.asStateFlow()

    private val _biometricEnabled = MutableStateFlow(false)
    val biometricEnabled: StateFlow<Boolean> = _biometricEnabled.asStateFlow()

    fun setDarkMode(enabled: Boolean) {
        _isDarkMode.value = enabled
        // preferencesRepository.setDarkMode(enabled)
    }

    fun setNotifications(enabled: Boolean) {
        _notificationsEnabled.value = enabled
        // preferencesRepository.setNotifications(enabled)
    }

    fun setBiometric(enabled: Boolean) {
        _biometricEnabled.value = enabled
        // preferencesRepository.setBiometric(enabled)
    }

    fun logout() {
        viewModelScope.launch {
            // authRepository.logout()
        }
    }
}
