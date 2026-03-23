package io.symthaea.spore

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.res.Configuration
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/**
 * Bridges Android battery, thermal, and night mode state to the Spore engine.
 *
 * Usage:
 * ```kotlin
 * lifecycle.addObserver(SporeBatteryBridge(context, engine))
 * ```
 */
class SporeBatteryBridge(
    private val context: Context,
    private val engine: SporeEngine,
) : DefaultLifecycleObserver {

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            val percent = if (scale > 0) (level * 100 / scale) else 50
            val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING
                    || status == BatteryManager.BATTERY_STATUS_FULL
            engine.setBatteryState(percent, isCharging)
        }
    }

    private var thermalListener: Any? = null // PowerManager.OnThermalStatusChangedListener (API 29+)

    override fun onStart(owner: LifecycleOwner) {
        // Battery
        context.registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

        // Thermal (API 29+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            val listener = PowerManager.OnThermalStatusChangedListener { status ->
                engine.setThermalLevel(status.coerceIn(0, 4))
            }
            pm.addThermalStatusListener(listener)
            thermalListener = listener
        }

        // Night mode
        updateNightMode()
    }

    override fun onStop(owner: LifecycleOwner) {
        context.unregisterReceiver(batteryReceiver)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
            @Suppress("UNCHECKED_CAST")
            (thermalListener as? PowerManager.OnThermalStatusChangedListener)?.let {
                pm.removeThermalStatusListener(it)
            }
            thermalListener = null
        }
    }

    override fun onResume(owner: LifecycleOwner) {
        updateNightMode()
    }

    private fun updateNightMode() {
        val nightFlags = context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK
        engine.setNightMode(nightFlags == Configuration.UI_MODE_NIGHT_YES)
    }
}
