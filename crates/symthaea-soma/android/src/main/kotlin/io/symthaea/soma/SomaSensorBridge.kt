package io.symthaea.soma

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/**
 * Bridges Android sensor data to the Soma engine.
 *
 * Registers for accelerometer, light, proximity, pressure, gyroscope,
 * and step counter sensors. Batches readings and forwards to
 * [SomaEngine.setSensors] at each tick.
 *
 * Lifecycle-aware: sensors register on onResume and unregister on onPause,
 * surviving Activity pause/resume cycles. Falls back to onStart/onStop
 * for Service contexts. Handles null SensorManager gracefully on devices
 * where the sensor service is unavailable.
 *
 * Usage:
 * ```kotlin
 * lifecycle.addObserver(SomaSensorBridge(context, engine))
 * ```
 */
class SomaSensorBridge(
    context: Context,
    private val engine: SomaEngine,
) : DefaultLifecycleObserver, SensorEventListener {

    private val sensorManager: SensorManager? =
        context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager

    // Latest sensor values (updated by sensor callbacks)
    @Volatile private var accelMagnitude: Float = 0f
    @Volatile private var lightLux: Float = 300f
    @Volatile private var proximityNear: Boolean = false
    @Volatile private var barometerHpa: Float = 1013.25f
    @Volatile private var rotationRate: Float = 0f
    @Volatile private var stepDelta: Int = 0
    private var lastStepCount: Int = -1

    /** Whether sensors are currently registered. Guards against double-register. */
    @Volatile private var registered: Boolean = false

    /** GPS novelty (0-1). Set externally from location updates. */
    @Volatile var gpsNovelty: Float = 0f

    override fun onResume(owner: LifecycleOwner) {
        registerAllSensors()
    }

    override fun onPause(owner: LifecycleOwner) {
        unregisterAllSensors()
    }

    // Fallback for Service-based lifecycle (Services emit onStart/onStop, not onResume/onPause)
    override fun onStart(owner: LifecycleOwner) {
        registerAllSensors()
    }

    override fun onStop(owner: LifecycleOwner) {
        unregisterAllSensors()
    }

    /** Push current sensor state to the engine. Call this each tick. */
    fun tick() {
        engine.setSensors(accelMagnitude, lightLux, proximityNear, barometerHpa, gpsNovelty)
        engine.setGyroscope(rotationRate)
        // Forward step delta and reset
        if (stepDelta > 0) {
            engine.setStepDelta(stepDelta)
            stepDelta = 0
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val (x, y, z) = event.values
                accelMagnitude = kotlin.math.sqrt(x * x + y * y + z * z) - 9.81f
                if (accelMagnitude < 0f) accelMagnitude = 0f
            }
            Sensor.TYPE_LIGHT -> lightLux = event.values[0]
            Sensor.TYPE_PROXIMITY -> {
                proximityNear = event.values[0] < (event.sensor.maximumRange * 0.5f)
            }
            Sensor.TYPE_PRESSURE -> barometerHpa = event.values[0]
            Sensor.TYPE_GYROSCOPE -> {
                val (gx, gy, gz) = event.values
                rotationRate = kotlin.math.sqrt(gx * gx + gy * gy + gz * gz)
            }
            Sensor.TYPE_STEP_COUNTER -> {
                val steps = event.values[0].toInt()
                if (lastStepCount >= 0) {
                    stepDelta += (steps - lastStepCount).coerceAtLeast(0)
                }
                lastStepCount = steps
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun registerAllSensors() {
        if (registered) return
        val sm = sensorManager
        if (sm == null) {
            Log.w(TAG, "SensorManager unavailable — sensor bridge disabled")
            return
        }
        registerSensor(sm, Sensor.TYPE_ACCELEROMETER)
        registerSensor(sm, Sensor.TYPE_LIGHT)
        registerSensor(sm, Sensor.TYPE_PROXIMITY)
        registerSensor(sm, Sensor.TYPE_PRESSURE)
        registerSensor(sm, Sensor.TYPE_GYROSCOPE)
        registerSensor(sm, Sensor.TYPE_STEP_COUNTER)
        registered = true
    }

    private fun unregisterAllSensors() {
        if (!registered) return
        sensorManager?.unregisterListener(this)
        registered = false
    }

    private fun registerSensor(sm: SensorManager, type: Int) {
        sm.getDefaultSensor(type)?.let {
            sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    companion object {
        private const val TAG = "SomaSensor"
    }
}
