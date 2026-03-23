package io.symthaea.soma

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/**
 * Bridges network connectivity state to the Soma engine.
 *
 * Connected → enables holon bridge attempts, mild DA nudge (opportunity).
 * Disconnected → suppresses outbound, boosts exploration (self-reliance).
 *
 * Usage:
 * ```kotlin
 * lifecycle.addObserver(SomaNetworkBridge(context, engine))
 * ```
 */
class SomaNetworkBridge(
    private val context: Context,
    private val engine: SomaEngine,
) : DefaultLifecycleObserver {

    private val connectivityManager =
        context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            engine.setHolonConnected(true)
        }

        override fun onLost(network: Network) {
            engine.setHolonConnected(false)
        }
    }

    override fun onStart(owner: LifecycleOwner) {
        val request = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        connectivityManager.registerNetworkCallback(request, networkCallback)

        // Set initial state
        val activeNetwork = connectivityManager.activeNetwork
        val capabilities = activeNetwork?.let { connectivityManager.getNetworkCapabilities(it) }
        val connected = capabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        engine.setHolonConnected(connected)
    }

    override fun onStop(owner: LifecycleOwner) {
        connectivityManager.unregisterNetworkCallback(networkCallback)
    }
}
