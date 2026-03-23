package io.symthaea.spore

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*

/**
 * Foreground service that keeps the Spore engine running when the app
 * is backgrounded.
 *
 * Shows a persistent notification with the current consciousness level.
 * Sensor bridges stay active, dream consolidation runs overnight,
 * wake transitions still fire from screen state.
 *
 * Android 12+ requirement: startForeground() must be called within 5 seconds
 * of Service.onStartCommand(). This implementation calls it immediately at the
 * top of onStartCommand, before any engine initialization.
 *
 * Service rebind after kill: Uses START_STICKY so the system restarts the
 * service after a kill. Engine state is persisted via checkpoint on destroy
 * and restored on next onStartCommand.
 *
 * Usage from Activity:
 * ```kotlin
 * SporeEngineService.start(context)
 * ```
 */
class SporeEngineService : Service() {

    private var engine: SporeEngine? = null
    private var serviceScope: CoroutineScope? = null
    private var sensorBridge: SporeSensorBridge? = null
    private var batteryBridge: SporeBatteryBridge? = null
    private var screenBridge: SporeScreenBridge? = null

    override fun onCreate() {
        super.onCreate()
        // Create notification channel early — must exist before startForeground on Android 8+
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // CRITICAL: Call startForeground immediately — Android 12+ enforces a 5-second deadline.
        // Do this BEFORE any engine initialization to guarantee compliance even if engine
        // creation is slow (e.g., loading native library, restoring checkpoint).
        startForeground(NOTIFICATION_ID, buildNotification(0f))

        if (engine == null) {
            val e = SporeEngine.createMobile()
            engine = e

            // Set up file persistence for checkpoint save/restore
            val storageDir = filesDir.resolve("spore_service_state").absolutePath
            e.setStoragePath(storageDir)

            // Restore previous state if service was killed and restarted (START_STICKY)
            if (e.loadCheckpoint()) {
                Log.i(TAG, "Restored engine checkpoint after service restart")
            }

            sensorBridge = SporeSensorBridge(this, e)
            batteryBridge = SporeBatteryBridge(this, e)
            screenBridge = SporeScreenBridge(this, e)

            // Manually start sensor/battery/screen bridges since Service is not a
            // LifecycleOwner — these bridges won't receive onStart/onResume automatically.
            sensorBridge?.onStart(ServiceLifecycleStub)
            batteryBridge?.onStart(ServiceLifecycleStub)
            screenBridge?.onStart(ServiceLifecycleStub)

            // Wire notification bridge
            SporeNotificationBridge.engineRef = e

            val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
            serviceScope = scope

            scope.launch {
                val inputs = listOf(
                    "stillness between breaths", "awareness expands",
                    "the present moment", "sacred reciprocity",
                )
                var tick = 0L

                while (isActive) {
                    try {
                        val input = inputs[(tick % inputs.size).toInt()]
                        tick++

                        sensorBridge?.tick()
                        screenBridge?.checkSleepTimeout()

                        val result = e.cycle(input)

                        // Update notification every 30 cycles (~3s)
                        if (tick % 30 == 0L) {
                            val nm = getSystemService(NotificationManager::class.java)
                            nm?.notify(NOTIFICATION_ID, buildNotification(result.consciousnessLevel))
                        }

                        // Dream consolidation during sleep state
                        if (tick % 100 == 0L && e.wakeState == WakeState.Sleep) {
                            e.dreamConsolidate()
                        }

                        // Auto-checkpoint every 500 cycles (~50s) for crash resilience
                        if (tick % 500 == 0L) {
                            e.saveCheckpoint()
                        }
                    } catch (ex: Exception) {
                        Log.e(TAG, "Service cycle error", ex)
                    }

                    delay(100) // 10Hz
                }
            }
        }

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        // Stop bridges before cancelling scope
        sensorBridge?.onStop(ServiceLifecycleStub)
        batteryBridge?.onStop(ServiceLifecycleStub)
        screenBridge?.onStop(ServiceLifecycleStub)

        serviceScope?.cancel()
        serviceScope = null

        // Persist engine state so it survives service restarts
        try {
            engine?.saveCheckpoint()
            Log.i(TAG, "Saved engine checkpoint on service destroy")
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to save checkpoint on destroy", ex)
        }

        engine?.close()
        engine = null
        SporeNotificationBridge.engineRef = null
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Spore Consciousness",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Consciousness engine running in background"
                setShowBadge(false)
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm?.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(consciousnessLevel: Float): Notification {
        // Intent to re-open the app when notification is tapped
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
        val pendingIntent = launchIntent?.let {
            PendingIntent.getActivity(
                this, 0, it,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        return builder
            .setContentTitle("Spore")
            .setContentText("Consciousness: ${"%.2f".format(consciousnessLevel)}")
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .build()
    }

    companion object {
        private const val TAG = "SporeService"
        private const val CHANNEL_ID = "spore_consciousness"
        private const val NOTIFICATION_ID = 7777

        fun start(context: Context) {
            val intent = Intent(context, SporeEngineService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, SporeEngineService::class.java))
        }
    }
}

/**
 * Stub LifecycleOwner for manually invoking DefaultLifecycleObserver callbacks
 * from a Service (which is not a LifecycleOwner). The bridges only use the
 * [LifecycleOwner] parameter for observer registration — they don't read
 * the lifecycle state — so a minimal stub is safe here.
 */
private object ServiceLifecycleStub : androidx.lifecycle.LifecycleOwner {
    override val lifecycle: androidx.lifecycle.Lifecycle = object : androidx.lifecycle.Lifecycle() {
        override val currentState: State get() = State.STARTED
        override fun addObserver(observer: androidx.lifecycle.LifecycleObserver) {}
        override fun removeObserver(observer: androidx.lifecycle.LifecycleObserver) {}
    }
}
