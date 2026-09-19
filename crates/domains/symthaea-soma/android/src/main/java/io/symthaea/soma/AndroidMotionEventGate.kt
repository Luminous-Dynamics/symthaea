package io.symthaea.soma

import android.os.Build
import android.view.MotionEvent
import android.view.View

/**
 * QUAL-ANDROIDMOTION-525 Android-side gate for assurance-bearing touch input.
 *
 * This gate rejects MotionEvents that are fully obscured, partially obscured,
 * multi-pointer, out of the protected view bounds, non-finite, or outside the
 * closed DOWN/MOVE/UP/CANCEL action model before invoking QUAL-ANDROIDINGRESS-522.
 *
 * `MotionEvent.eventTime` is forwarded as Android uptime-based data. It is not a
 * wall-clock or causal-order proof.
 */
class AndroidMotionEventGate(
    private val protectedView: View,
    private val attentionGuard: AndroidAttentionGuard,
    private val rejectPartialObscuration: Boolean = true,
) {
    enum class RejectReason {
        PARTIAL_OBSCURATION_UNOBSERVABLE,
        WINDOW_OBSCURED,
        WINDOW_PARTIALLY_OBSCURED,
        MULTI_POINTER,
        UNSUPPORTED_ACTION,
        INVALID_VIEW_BOUNDS,
        NON_FINITE_VALUE,
        OUT_OF_RANGE_VALUE,
        NON_POSITIVE_SEQUENCE_OR_GENERATION,
    }

    data class MotionEvidence(
        val sdkInt: Int,
        val eventFlags: Int,
        val pointerCount: Int,
        val action: Int,
        val normalizedX: Float,
        val normalizedY: Float,
        val pressure: Float,
        val eventTimeMs: Long,
        val attentionSnapshot: AndroidAttentionGuard.AttentionSnapshot,
    )

    sealed class GateResult {
        data class Accepted(
            val motion: MotionEvidence,
            val ingress: AssuranceNativeBindings.TouchIngressResult,
        ) : GateResult()

        data class NativeIngressRejected(
            val motion: MotionEvidence,
            val ingress: AssuranceNativeBindings.TouchIngressResult,
        ) : GateResult()

        data class Rejected(val reason: RejectReason) : GateResult()
    }

    fun dispatchTouchV1(
        event: MotionEvent,
        handle: Long,
        abiVersion: Int,
        maxFrameBytes: Long,
        captureProfileId: ByteArray,
        touchInputProfileId: ByteArray,
        inputAttesterId: ByteArray,
        interactionGeneration: Long,
        inputSequence: Long,
    ): GateResult {
        if (interactionGeneration <= 0 || inputSequence <= 0) {
            return GateResult.Rejected(RejectReason.NON_POSITIVE_SEQUENCE_OR_GENERATION)
        }

        if (rejectPartialObscuration && Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return GateResult.Rejected(RejectReason.PARTIAL_OBSCURATION_UNOBSERVABLE)
        }

        val flags = event.flags
        if ((flags and MotionEvent.FLAG_WINDOW_IS_OBSCURED) != 0) {
            return GateResult.Rejected(RejectReason.WINDOW_OBSCURED)
        }
        if (
            rejectPartialObscuration &&
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
            (flags and MotionEvent.FLAG_WINDOW_IS_PARTIALLY_OBSCURED) != 0
        ) {
            return GateResult.Rejected(RejectReason.WINDOW_PARTIALLY_OBSCURED)
        }

        if (event.pointerCount != 1) {
            return GateResult.Rejected(RejectReason.MULTI_POINTER)
        }

        val action = when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> 0
            MotionEvent.ACTION_MOVE -> 1
            MotionEvent.ACTION_UP -> 2
            MotionEvent.ACTION_CANCEL -> 3
            else -> return GateResult.Rejected(RejectReason.UNSUPPORTED_ACTION)
        }

        val width = protectedView.width
        val height = protectedView.height
        if (width <= 0 || height <= 0) {
            return GateResult.Rejected(RejectReason.INVALID_VIEW_BOUNDS)
        }

        val normalizedX = event.x / width.toFloat()
        val normalizedY = event.y / height.toFloat()
        val pressure = event.pressure
        if (!normalizedX.isFinite() || !normalizedY.isFinite() || !pressure.isFinite()) {
            return GateResult.Rejected(RejectReason.NON_FINITE_VALUE)
        }
        if (
            normalizedX !in 0.0f..1.0f ||
            normalizedY !in 0.0f..1.0f ||
            pressure !in 0.0f..1.0f
        ) {
            return GateResult.Rejected(RejectReason.OUT_OF_RANGE_VALUE)
        }

        val attentionSnapshot = attentionGuard.snapshot()
        val motion = MotionEvidence(
            sdkInt = Build.VERSION.SDK_INT,
            eventFlags = flags,
            pointerCount = event.pointerCount,
            action = action,
            normalizedX = normalizedX,
            normalizedY = normalizedY,
            pressure = pressure,
            eventTimeMs = event.eventTime,
            attentionSnapshot = attentionSnapshot,
        )

        val ingress = AssuranceNativeBindings.touchEventV1(
            handle = handle,
            abiVersion = abiVersion,
            maxFrameBytes = maxFrameBytes,
            captureProfileId = captureProfileId,
            touchInputProfileId = touchInputProfileId,
            inputAttesterId = inputAttesterId,
            interactionGeneration = interactionGeneration,
            inputSequence = inputSequence,
            x = normalizedX,
            y = normalizedY,
            action = action,
            pressure = pressure,
            timestampMs = event.eventTime,
        )

        return if (ingress.accepted) {
            GateResult.Accepted(motion = motion, ingress = ingress)
        } else {
            GateResult.NativeIngressRejected(motion = motion, ingress = ingress)
        }
    }
}
