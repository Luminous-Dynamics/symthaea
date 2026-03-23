package io.symthaea.spore

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/**
 * Bridges screen on/off state to the Spore engine as wake/sleep signals.
 *
 * Screen on → PhonePickup wake signal (most natural wake trigger).
 * Screen off → after 60s of sustained off, sends ExplicitSleep.
 *
 * Usage:
 * ```kotlin
 * lifecycle.addObserver(SporeScreenBridge(context, engine))
 * ```
 */
class SporeScreenBridge(
    private val context: Context,
    private val engine: SporeEngine,
) : DefaultLifecycleObserver {

    @Volatile private var screenOffSince: Long = 0L

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            when (intent.action) {
                Intent.ACTION_SCREEN_ON -> {
                    screenOffSince = 0L
                    engine.sendWakeSignal(WakeSignal.PhonePickup)
                }
                Intent.ACTION_SCREEN_OFF -> {
                    screenOffSince = System.currentTimeMillis()
                }
            }
        }
    }

    override fun onStart(owner: LifecycleOwner) {
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
        }
        context.registerReceiver(screenReceiver, filter)
    }

    override fun onStop(owner: LifecycleOwner) {
        context.unregisterReceiver(screenReceiver)
    }

    /**
     * Check if screen has been off long enough to trigger sleep.
     * Call periodically (e.g., each engine tick).
     */
    fun checkSleepTimeout() {
        val off = screenOffSince
        if (off > 0L && System.currentTimeMillis() - off > SLEEP_TIMEOUT_MS) {
            engine.sendWakeSignal(WakeSignal.ExplicitSleep)
            screenOffSince = 0L // Don't re-trigger
        }
    }

    companion object {
        private const val SLEEP_TIMEOUT_MS = 60_000L
    }
}
