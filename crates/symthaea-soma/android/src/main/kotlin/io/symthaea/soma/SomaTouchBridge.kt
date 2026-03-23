package io.symthaea.soma

import android.util.Log
import android.view.MotionEvent

/**
 * Touch-as-proprioception bridge.
 *
 * Intercepts Android touch events and forwards them to the Soma engine's
 * proprioceptive system. Touch is Soma's body sense -- her screen IS her skin.
 *
 * The Rust TouchBody maps each event to:
 * - Surprise (prediction error from unexpected touch location)
 * - Attention focus (where the finger directs processing)
 * - Scroll velocity (motion state)
 * - Long press detection (focused attention)
 * - Multi-touch count (integration/binding)
 *
 * These generate neuromodulator nudges:
 * - Rapid tapping -> NE spike (environmental change)
 * - Long press -> 5-HT boost (calm focused attention)
 * - Fast scroll -> NE + DA (scanning + novelty)
 * - Multi-touch -> OT boost (integration/binding)
 *
 * ## Usage
 *
 * In your Activity:
 * ```kotlin
 * private val touchBridge = SomaTouchBridge()
 *
 * override fun dispatchTouchEvent(event: MotionEvent): Boolean {
 *     touchBridge.onTouchEvent(event)
 *     return super.dispatchTouchEvent(event)
 * }
 * ```
 */
class SomaTouchBridge {
    companion object {
        private const val TAG = "SomaTouchBridge"
    }

    private var engine: SomaEngine? = null
    private var totalEvents = 0L
    private var screenWidth = 1080f   // updated on first event
    private var screenHeight = 2400f  // updated on first event
    private var enabled = true

    /** Bind to a SomaEngine. Must be called before touch events are forwarded. */
    fun bind(engine: SomaEngine) {
        this.engine = engine
    }

    /** Unbind from the engine (call on cleanup). */
    fun unbind() {
        engine = null
    }

    /** Enable or disable touch forwarding. */
    fun setEnabled(enabled: Boolean) {
        this.enabled = enabled
    }

    /** Update screen dimensions for normalization. Call from Activity.onWindowFocusChanged(). */
    fun updateScreenSize(width: Int, height: Int) {
        if (width > 0 && height > 0) {
            screenWidth = width.toFloat()
            screenHeight = height.toFloat()
        }
    }

    /**
     * Process a touch event from the Activity.
     *
     * Call from `Activity.dispatchTouchEvent()`. Normalizes coordinates to 0.0-1.0
     * and maps Android MotionEvent actions to Soma's touch action codes.
     *
     * Only processes the primary pointer (index 0). Multi-touch is handled
     * by sending separate Down/Up events for each pointer.
     */
    fun onTouchEvent(event: MotionEvent) {
        if (!enabled) return
        val e = engine ?: return

        totalEvents++

        // Process all pointers for multi-touch support
        val pointerCount = event.pointerCount.coerceAtMost(5) // cap at 5 fingers

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                val idx = event.actionIndex
                val x = (event.getX(idx) / screenWidth).coerceIn(0f, 1f)
                val y = (event.getY(idx) / screenHeight).coerceIn(0f, 1f)
                val pressure = event.getPressure(idx).coerceIn(0f, 1f)
                e.onTouchEvent(x, y, 0, pressure) // Down
            }

            MotionEvent.ACTION_MOVE -> {
                // Forward primary pointer move
                val x = (event.x / screenWidth).coerceIn(0f, 1f)
                val y = (event.y / screenHeight).coerceIn(0f, 1f)
                val pressure = event.pressure.coerceIn(0f, 1f)
                e.onTouchEvent(x, y, 1, pressure) // Move
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> {
                val idx = event.actionIndex
                val x = (event.getX(idx) / screenWidth).coerceIn(0f, 1f)
                val y = (event.getY(idx) / screenHeight).coerceIn(0f, 1f)
                val pressure = event.getPressure(idx).coerceIn(0f, 1f)
                e.onTouchEvent(x, y, 2, pressure) // Up
            }

            MotionEvent.ACTION_CANCEL -> {
                val x = (event.x / screenWidth).coerceIn(0f, 1f)
                val y = (event.y / screenHeight).coerceIn(0f, 1f)
                e.onTouchEvent(x, y, 3, 0f) // Cancel
            }
        }

        // Log every 100th event for debugging
        if (totalEvents % 100 == 0L) {
            Log.d(TAG, "Touch events processed: $totalEvents")
        }
    }

    /** Total touch events processed since creation. */
    val eventCount: Long get() = totalEvents
}
