package io.symthaea.soma

/**
 * QUAL-ANDROIDINGRESS-522 Kotlin façade for the checked native assurance ingress.
 *
 * This API is intentionally separate from legacy [NativeBindings]. It allocates
 * fixed-size output buffers so callers cannot vary the result shape, and it
 * returns the exact profile / observation / ingress-receipt identities emitted by
 * QUAL-NATIVEINGRESS-521.
 */
object AssuranceNativeBindings {
    const val STATUS_ACCEPTED: Int = 0
    const val JNI_ARGUMENT_ERROR: Int = 1000
    const val JNI_ACCESS_ERROR: Int = 1001

    const val PLATFORM_ANDROID_JNI: Int = 1
    const val PLATFORM_IOS_C_ABI: Int = 2

    private const val ID_LEN = 32
    private const val OUTPUT_IDS_LEN = ID_LEN * 3

    init {
        System.loadLibrary("soma_jni")
    }

    data class AssuranceIds(
        val profileId: ByteArray,
        val observationId: ByteArray,
        val ingressReceiptId: ByteArray,
    ) {
        init {
            require(profileId.size == ID_LEN)
            require(observationId.size == ID_LEN)
            require(ingressReceiptId.size == ID_LEN)
        }
    }

    data class FrameIngressResult(
        val status: Int,
        val ids: AssuranceIds,
        val surprise: Float,
    ) {
        val accepted: Boolean get() = status == STATUS_ACCEPTED
    }

    data class TouchIngressResult(
        val status: Int,
        val ids: AssuranceIds,
    ) {
        val accepted: Boolean get() = status == STATUS_ACCEPTED
    }

    @JvmStatic
    fun injectFrameV1(
        handle: Long,
        data: ByteArray,
        width: Int,
        height: Int,
        channels: Int,
        abiVersion: Int,
        maxFrameBytes: Long,
        captureProfileId: ByteArray,
        touchInputProfileId: ByteArray,
        surfaceId: ByteArray,
        surfaceGeneration: Long,
        interactionGeneration: Long,
        frameSequence: Long,
    ): FrameIngressResult {
        require(captureProfileId.size == ID_LEN)
        require(touchInputProfileId.size == ID_LEN)
        require(surfaceId.size == ID_LEN)

        val ids = ByteArray(OUTPUT_IDS_LEN)
        val surprise = FloatArray(1)
        val status = injectFrameV1Native(
            handle = handle,
            data = data,
            width = width,
            height = height,
            channels = channels,
            platform = PLATFORM_ANDROID_JNI,
            abiVersion = abiVersion,
            maxFrameBytes = maxFrameBytes,
            captureProfileId = captureProfileId,
            touchInputProfileId = touchInputProfileId,
            surfaceId = surfaceId,
            surfaceGeneration = surfaceGeneration,
            interactionGeneration = interactionGeneration,
            frameSequence = frameSequence,
            outIds = ids,
            outSurprise = surprise,
        )
        return FrameIngressResult(
            status = status,
            ids = decodeIds(ids),
            surprise = surprise[0],
        )
    }

    @JvmStatic
    fun touchEventV1(
        handle: Long,
        abiVersion: Int,
        maxFrameBytes: Long,
        captureProfileId: ByteArray,
        touchInputProfileId: ByteArray,
        inputAttesterId: ByteArray,
        interactionGeneration: Long,
        inputSequence: Long,
        x: Float,
        y: Float,
        action: Int,
        pressure: Float,
        timestampMs: Long,
    ): TouchIngressResult {
        require(captureProfileId.size == ID_LEN)
        require(touchInputProfileId.size == ID_LEN)
        require(inputAttesterId.size == ID_LEN)

        val ids = ByteArray(OUTPUT_IDS_LEN)
        val status = touchEventV1Native(
            handle = handle,
            platform = PLATFORM_ANDROID_JNI,
            abiVersion = abiVersion,
            maxFrameBytes = maxFrameBytes,
            captureProfileId = captureProfileId,
            touchInputProfileId = touchInputProfileId,
            inputAttesterId = inputAttesterId,
            interactionGeneration = interactionGeneration,
            inputSequence = inputSequence,
            x = x,
            y = y,
            action = action,
            pressure = pressure,
            timestampMs = timestampMs,
            outIds = ids,
        )
        return TouchIngressResult(status = status, ids = decodeIds(ids))
    }

    private fun decodeIds(bytes: ByteArray): AssuranceIds {
        require(bytes.size == OUTPUT_IDS_LEN)
        return AssuranceIds(
            profileId = bytes.copyOfRange(0, ID_LEN),
            observationId = bytes.copyOfRange(ID_LEN, ID_LEN * 2),
            ingressReceiptId = bytes.copyOfRange(ID_LEN * 2, OUTPUT_IDS_LEN),
        )
    }

    @JvmStatic
    private external fun injectFrameV1Native(
        handle: Long,
        data: ByteArray,
        width: Int,
        height: Int,
        channels: Int,
        platform: Int,
        abiVersion: Int,
        maxFrameBytes: Long,
        captureProfileId: ByteArray,
        touchInputProfileId: ByteArray,
        surfaceId: ByteArray,
        surfaceGeneration: Long,
        interactionGeneration: Long,
        frameSequence: Long,
        outIds: ByteArray,
        outSurprise: FloatArray,
    ): Int

    @JvmStatic
    private external fun touchEventV1Native(
        handle: Long,
        platform: Int,
        abiVersion: Int,
        maxFrameBytes: Long,
        captureProfileId: ByteArray,
        touchInputProfileId: ByteArray,
        inputAttesterId: ByteArray,
        interactionGeneration: Long,
        inputSequence: Long,
        x: Float,
        y: Float,
        action: Int,
        pressure: Float,
        timestampMs: Long,
        outIds: ByteArray,
    ): Int
}
