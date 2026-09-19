package io.symthaea.soma

import android.app.Activity
import android.os.Build
import android.view.View
import android.view.WindowManager

/**
 * QUAL-ANDROIDATTENTION-524 Android-side collector/configurator.
 *
 * This class deliberately exposes a vector of platform facts rather than one
 * "secure attention" boolean. In particular, successfully requesting
 * hide-overlay-windows is recorded as a request fact, not proof that no trusted
 * system/accessibility/IME overlay exists.
 */
class AndroidAttentionGuard(
    private val activity: Activity,
    private val protectedView: View,
) {
    private var hideApplicationOverlaysRequested: Boolean = false

    data class ProtectionApplication(
        val flagSecureSet: Boolean,
        val hideApplicationOverlaysRequested: Boolean,
        val filterTouchesWhenObscuredEnabled: Boolean,
    )

    data class AttentionSnapshot(
        val sdkInt: Int,
        val windowHasFocus: Boolean,
        val flagSecureSet: Boolean,
        val hideApplicationOverlaysRequested: Boolean,
        val filterTouchesWhenObscuredEnabled: Boolean,
        val viewAttachedToWindow: Boolean,
        val viewShown: Boolean,
        val activityTopResumed: Boolean,
        val activityInMultiWindowMode: Boolean,
    )

    /**
     * Apply the protections owned by the application.
     *
     * `setHideOverlayWindows(true)` is available only on API 31+. A successful
     * call records that the protection was requested; it does not imply that
     * every possible system/trusted overlay class is absent.
     */
    fun applyProtections(
        setFlagSecure: Boolean = true,
        requestHideApplicationOverlays: Boolean = true,
        filterTouchesWhenObscured: Boolean = true,
    ): ProtectionApplication {
        if (setFlagSecure) {
            activity.window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
        }

        protectedView.filterTouchesWhenObscured = filterTouchesWhenObscured

        hideApplicationOverlaysRequested = false
        if (requestHideApplicationOverlays && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                activity.window.setHideOverlayWindows(true)
                hideApplicationOverlaysRequested = true
            } catch (_: SecurityException) {
                // Fail closed in the observation: the requested protection is
                // recorded as false and policy can reject the snapshot.
                hideApplicationOverlaysRequested = false
            }
        }

        return ProtectionApplication(
            flagSecureSet = isFlagSecureSet(),
            hideApplicationOverlaysRequested = hideApplicationOverlaysRequested,
            filterTouchesWhenObscuredEnabled = protectedView.filterTouchesWhenObscured,
        )
    }

    /**
     * Stop requesting application-overlay hiding when the protected flow ends.
     * Other protections are intentionally not silently removed here.
     */
    fun releaseOverlayHiding() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && hideApplicationOverlaysRequested) {
            activity.window.setHideOverlayWindows(false)
        }
        hideApplicationOverlaysRequested = false
    }

    /** Capture current facts for policy evaluation. */
    fun snapshot(): AttentionSnapshot {
        val topResumed = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            activity.isTopResumedActivity
        } else {
            false
        }

        return AttentionSnapshot(
            sdkInt = Build.VERSION.SDK_INT,
            windowHasFocus = activity.hasWindowFocus(),
            flagSecureSet = isFlagSecureSet(),
            hideApplicationOverlaysRequested = hideApplicationOverlaysRequested,
            filterTouchesWhenObscuredEnabled = protectedView.filterTouchesWhenObscured,
            viewAttachedToWindow = protectedView.isAttachedToWindow,
            viewShown = protectedView.isShown,
            activityTopResumed = topResumed,
            activityInMultiWindowMode = activity.isInMultiWindowMode,
        )
    }

    private fun isFlagSecureSet(): Boolean =
        (activity.window.attributes.flags and WindowManager.LayoutParams.FLAG_SECURE) != 0
}
