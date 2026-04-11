// Track CH: Android Native App (Kotlin)
// Mycelix Pulse - Decentralized communication with Web-of-Trust

package com.mycelix.mail

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.mycelix.mail.ui.theme.MycelixMailTheme
import com.mycelix.mail.ui.screens.*
import com.mycelix.mail.ui.viewmodels.*
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MycelixMailTheme {
                MycelixMailApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MycelixMailApp() {
    val navController = rememberNavController()
    val authViewModel: AuthViewModel = viewModel()
    val isAuthenticated by authViewModel.isAuthenticated.collectAsState()

    if (isAuthenticated) {
        MainScaffold(navController = navController)
    } else {
        LoginScreen(
            onLoginSuccess = { navController.navigate("inbox") },
            viewModel = authViewModel
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScaffold(navController: NavHostController) {
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route ?: "inbox"

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Inbox, contentDescription = "Inbox") },
                    label = { Text("Inbox") },
                    selected = currentRoute == "inbox",
                    onClick = { navController.navigate("inbox") }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Group, contentDescription = "Trust") },
                    label = { Text("Trust") },
                    selected = currentRoute == "trust",
                    onClick = { navController.navigate("trust") }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Edit, contentDescription = "Compose") },
                    label = { Text("Compose") },
                    selected = currentRoute == "compose",
                    onClick = { navController.navigate("compose") }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Search, contentDescription = "Search") },
                    label = { Text("Search") },
                    selected = currentRoute == "search",
                    onClick = { navController.navigate("search") }
                )
                NavigationBarItem(
                    icon = { Icon(Icons.Filled.Settings, contentDescription = "Settings") },
                    label = { Text("Settings") },
                    selected = currentRoute == "settings",
                    onClick = { navController.navigate("settings") }
                )
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = "inbox",
            modifier = Modifier.padding(paddingValues)
        ) {
            composable("inbox") {
                InboxScreen(
                    onEmailClick = { emailId ->
                        navController.navigate("email/$emailId")
                    }
                )
            }
            composable("email/{emailId}") { backStackEntry ->
                val emailId = backStackEntry.arguments?.getString("emailId") ?: return@composable
                EmailDetailScreen(
                    emailId = emailId,
                    onBack = { navController.popBackStack() }
                )
            }
            composable("trust") {
                TrustNetworkScreen()
            }
            composable("compose") {
                ComposeScreen(
                    onSent = { navController.navigate("inbox") }
                )
            }
            composable("search") {
                SearchScreen(
                    onEmailClick = { emailId ->
                        navController.navigate("email/$emailId")
                    }
                )
            }
            composable("settings") {
                SettingsScreen()
            }
        }
    }
}
