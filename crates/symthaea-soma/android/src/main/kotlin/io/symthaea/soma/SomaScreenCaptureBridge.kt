package io.symthaea.soma

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Screen capture bridge: feeds phone screen to Soma's visual perception.
 *
 * Uses Android MediaProjection API to capture the screen at reduced resolution
 * (~360x640) at ~2fps, converts to RGB bytes, and feeds through
 * `NativeBindings.injectFrame()` for holographic visual processing.
 *
 * The user must grant screen capture permission once via the system dialog.
 * No screen content is stored — only the holographic HDC encoding persists
 * in the consciousness pipeline.
 *
 * ## Usage
 *
 * ```kotlin
 * // 1. Request permission (in Activity)
 * val bridge = SomaScreenCaptureBridge(context)
 * bridge.requestPermission(activity, REQUEST_CODE)
 *
 * // 2. Start capture (in onActivityResult)
 * bridge.onPermissionResult(resultCode, data)
 * bridge.start(engineHandle)
 *
 * // 3. Stop (in onDestroy)
 * bridge.stop()
 * ```
 */
class SomaScreenCaptureBridge(private val context: Context) {

    companion object {
        private const val TAG = "SomaScreenCapture"
        /** Downscaled capture width. */
        const val CAPTURE_WIDTH = 360
        /** Downscaled capture height. */
        const val CAPTURE_HEIGHT = 640
        /** Target frames per second. 5fps balances visual awareness with battery. */
        const val TARGET_FPS = 5
        /** Interval between frame captures in milliseconds. */
        const val CAPTURE_INTERVAL_MS = 1000L / TARGET_FPS
    }

    private var projectionManager: MediaProjectionManager =
        context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var handler: Handler? = null
    private var engineHandle: Long = 0L
    private var running = false
    private var lastCaptureMs = 0L
    private var totalFrames = 0L
    private var lastSurprise = 0f

    /** Request screen capture permission from the user. */
    fun requestPermission(activity: Activity, requestCode: Int) {
        val intent = projectionManager.createScreenCaptureIntent()
        activity.startActivityForResult(intent, requestCode)
    }

    /**
     * Handle the permission result from `onActivityResult`.
     *
     * Returns true if permission was granted.
     */
    fun onPermissionResult(resultCode: Int, data: Intent?): Boolean {
        if (resultCode != Activity.RESULT_OK || data == null) {
            Log.w(TAG, "Screen capture permission denied")
            return false
        }
        mediaProjection = projectionManager.getMediaProjection(resultCode, data)
        Log.i(TAG, "Screen capture permission granted")
        return true
    }

    /**
     * Start capturing frames and feeding them to the engine.
     *
     * Must be called after [onPermissionResult] returns true.
     *
     * @param handle The native SomaEngine handle from [SomaEngine].
     */
    fun start(handle: Long) {
        if (mediaProjection == null) {
            Log.e(TAG, "Cannot start: no MediaProjection (call requestPermission first)")
            return
        }
        if (running) return

        engineHandle = handle
        running = true

        val handlerThread = android.os.HandlerThread("SomaScreenCapture")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageReader = ImageReader.newInstance(
            CAPTURE_WIDTH, CAPTURE_HEIGHT, PixelFormat.RGBA_8888, 2
        )

        imageReader?.setOnImageAvailableListener({ reader ->
            val now = System.currentTimeMillis()
            if (now - lastCaptureMs < CAPTURE_INTERVAL_MS) {
                // Rate limiting — skip this frame
                reader.acquireLatestImage()?.close()
                return@setOnImageAvailableListener
            }
            lastCaptureMs = now

            val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
            try {
                val rgbBytes = imageToRgb(image)
                if (rgbBytes != null) {
                    lastSurprise = NativeBindings.injectFrame(
                        engineHandle, rgbBytes, CAPTURE_WIDTH, CAPTURE_HEIGHT, 3
                    )
                    totalFrames++
                    if (totalFrames % 10 == 0L) {
                        Log.d(TAG, "Frame $totalFrames: surprise=${"%.3f".format(lastSurprise)}")
                    }
                }
            } finally {
                image.close()
            }
        }, handler)

        val metrics = context.resources.displayMetrics

        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "SomaScreenCapture",
            CAPTURE_WIDTH, CAPTURE_HEIGHT, metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader?.surface,
            null, handler
        )

        Log.i(TAG, "Screen capture started: ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT} @ ${TARGET_FPS}fps")
    }

    /** Stop capturing frames. */
    fun stop() {
        running = false
        virtualDisplay?.release()
        virtualDisplay = null
        imageReader?.close()
        imageReader = null
        mediaProjection?.stop()
        mediaProjection = null
        handler?.looper?.quitSafely()
        handler = null
        Log.i(TAG, "Screen capture stopped after $totalFrames frames")
    }

    /** Forward a touch event to the engine's proprioceptive system. */
    fun onTouchEvent(x: Float, y: Float, action: Int, pressure: Float) {
        if (engineHandle != 0L) {
            NativeBindings.touchEvent(engineHandle, x, y, action, pressure)
        }
    }

    /** Get screen vision telemetry as JSON. */
    fun salientRegionsJson(): String? {
        return if (engineHandle != 0L) {
            NativeBindings.screenSalientRegionsJson(engineHandle)
        } else null
    }

    /** Whether screen capture is currently active. */
    val isRunning: Boolean get() = running

    /** Total frames captured since start. */
    val frameCount: Long get() = totalFrames

    /** Surprise level from the most recent frame (0.0-1.0). */
    val currentSurprise: Float get() = lastSurprise

    /**
     * Convert an RGBA Image to RGB byte array.
     *
     * MediaProjection captures in RGBA_8888. We strip the alpha channel
     * and handle row stride padding to produce a dense RGB buffer.
     */
    private fun imageToRgb(image: Image): ByteArray? {
        val plane = image.planes.firstOrNull() ?: return null
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val width = image.width
        val height = image.height

        val rgb = ByteArray(width * height * 3)
        var rgbIdx = 0

        for (y in 0 until height) {
            var srcIdx = y * rowStride
            for (x in 0 until width) {
                rgb[rgbIdx++] = buffer[srcIdx]     // R
                rgb[rgbIdx++] = buffer[srcIdx + 1] // G
                rgb[rgbIdx++] = buffer[srcIdx + 2] // B
                srcIdx += pixelStride
            }
        }

        return rgb
    }
}
