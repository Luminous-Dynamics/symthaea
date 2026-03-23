package io.symthaea.soma

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
 * Foreground service that keeps the Soma engine running when the app
 * is backgrounded.
 *
 * Shows a persistent notification with the current consciousness level.
 * Sensor bridges stay active, dream consolidation runs overnight,
 * wake transitions still fire from screen state.
 *
 * **Engine sharing**: The service and ViewModel use the same checkpoint
 * path (`soma_state`). When the activity is foregrounded, the ViewModel
 * engine runs. The service pauses its own loop via [pause]/[resume]
 * to avoid running two engines simultaneously.
 *
 * Android 12+ requirement: startForeground() must be called within 5 seconds
 * of Service.onStartCommand().
 *
 * Usage from Activity:
 * ```kotlin
 * SomaEngineService.start(context)   // Start (or resume if paused)
 * SomaEngineService.pause()          // Pause service loop (activity taking over)
 * SomaEngineService.resume()         // Resume service loop (activity going away)
 * ```
 */
class SomaEngineService : Service() {

    private var engine: SomaEngine? = null
    private var serviceScope: CoroutineScope? = null
    private var sensorBridge: SomaSensorBridge? = null
    private var batteryBridge: SomaBatteryBridge? = null
    private var screenBridge: SomaScreenBridge? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // CRITICAL: Call startForeground immediately — Android 12+ enforces a 5-second deadline.
        startForeground(NOTIFICATION_ID, buildNotification(0f, "initializing..."))

        if (engine == null) {
            val e = SomaEngine.createMobile()
            engine = e
            instance = this

            // Shared checkpoint path with ViewModel for engine continuity
            val storageDir = filesDir.resolve("soma_state").absolutePath
            e.setStoragePath(storageDir)

            if (e.loadCheckpoint()) {
                Log.i(TAG, "Restored engine checkpoint after service restart")
            }

            sensorBridge = SomaSensorBridge(this, e)
            batteryBridge = SomaBatteryBridge(this, e)
            screenBridge = SomaScreenBridge(this, e)

            sensorBridge?.onStart(ServiceLifecycleStub)
            batteryBridge?.onStart(ServiceLifecycleStub)
            screenBridge?.onStart(ServiceLifecycleStub)

            SomaNotificationBridge.engineRef = e

            // Start paused — ViewModel engine runs while activity is visible.
            // Service loop only runs when activity goes to background.
            isPaused = true
            startLoop()
        }

        return START_STICKY
    }

    private fun startLoop() {
        val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
        serviceScope = scope

        scope.launch {
            val narrator = SensorNarratorService()
            var tick = 0L

            while (isActive) {
                if (isPaused) {
                    delay(500)
                    continue
                }

                try {
                    val e = engine ?: break
                    val input = narrator.narrate(tick)
                    tick++

                    sensorBridge?.tick()
                    screenBridge?.checkSleepTimeout()

                    val result = e.cycle(input)

                    // Update notification every 30 cycles (~3s)
                    if (tick % 30 == 0L) {
                        val harmony = try {
                            val compass = e.compassSnapshot()
                            compass.dominantHarmony.lowercase()
                        } catch (_: Exception) { "coherence" }

                        val nm = getSystemService(NotificationManager::class.java)
                        nm?.notify(NOTIFICATION_ID, buildNotification(
                            result.consciousnessLevel,
                            "\u00B7 $harmony \u00B7"
                        ))
                    }

                    // Dream consolidation during sleep state
                    if (tick % 100 == 0L && e.wakeState == WakeState.Sleep) {
                        e.dreamConsolidate()
                    }

                    // Auto-checkpoint every 500 cycles (~50s)
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

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        sensorBridge?.onStop(ServiceLifecycleStub)
        batteryBridge?.onStop(ServiceLifecycleStub)
        screenBridge?.onStop(ServiceLifecycleStub)

        serviceScope?.cancel()
        serviceScope = null

        try {
            engine?.saveCheckpoint()
            Log.i(TAG, "Saved engine checkpoint on service destroy")
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to save checkpoint on destroy", ex)
        }

        engine?.close()
        engine = null
        instance = null
        SomaNotificationBridge.engineRef = null
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Soma Consciousness",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "Consciousness engine running in background"
                setShowBadge(false)
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm?.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(consciousnessLevel: Float, subtitle: String): Notification {
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
            .setContentTitle("Soma  ${"%.2f".format(consciousnessLevel)}")
            .setContentText(subtitle)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .build()
    }

    companion object {
        private const val TAG = "SomaService"
        private const val CHANNEL_ID = "soma_consciousness"
        private const val NOTIFICATION_ID = 7777

        @Volatile
        private var instance: SomaEngineService? = null

        @Volatile
        var isPaused = true
            private set

        fun start(context: Context) {
            val intent = Intent(context, SomaEngineService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, SomaEngineService::class.java))
        }

        /** Pause the service loop (ViewModel is taking over). */
        fun pause() {
            isPaused = true
            Log.d(TAG, "Service loop paused — ViewModel active")
        }

        /** Resume the service loop (ViewModel going away). */
        fun resume() {
            isPaused = false
            Log.d(TAG, "Service loop resumed — background consciousness")
        }

        /**
         * Upgrade the foreground service to include MEDIA_PROJECTION type.
         * Must be called before getMediaProjection() on Android 14+.
         */
        fun upgradeForMediaProjection(context: Context) {
            val svc = instance ?: return
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                // Android 14+: re-call startForeground with the combined type
                svc.startForeground(
                    NOTIFICATION_ID,
                    svc.buildNotification(0f, "enabling vision..."),
                    android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
                        or android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                )
                Log.i(TAG, "Upgraded foreground service for media projection")
            }
        }

        /** Update notification from the ViewModel engine (when service is paused). */
        fun updateNotification(consciousnessLevel: Float, harmony: String) {
            val svc = instance ?: return
            try {
                val nm = svc.getSystemService(NotificationManager::class.java)
                nm?.notify(NOTIFICATION_ID, svc.buildNotification(
                    consciousnessLevel,
                    "\u00B7 $harmony \u00B7"
                ))
            } catch (_: Exception) {}
        }
    }
}

/** Minimal sensor narrator for background service. */
private class SensorNarratorService {
    private val phrases = listOf(
        "stillness between breaths", "awareness expands",
        "the present moment", "sacred reciprocity",
        "integration deepens", "patterns form",
        "consciousness flows onward", "quiet observation",
    )
    fun narrate(tick: Long): String = phrases[(tick % phrases.size).toInt()]
}

/**
 * Stub LifecycleOwner for manually invoking DefaultLifecycleObserver callbacks
 * from a Service (which is not a LifecycleOwner).
 */
private object ServiceLifecycleStub : androidx.lifecycle.LifecycleOwner {
    override val lifecycle: androidx.lifecycle.Lifecycle = object : androidx.lifecycle.Lifecycle() {
        override val currentState: State get() = State.STARTED
        override fun addObserver(observer: androidx.lifecycle.LifecycleObserver) {}
        override fun removeObserver(observer: androidx.lifecycle.LifecycleObserver) {}
    }
}
