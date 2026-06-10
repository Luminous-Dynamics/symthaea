package io.symthaea.soma.demo

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.os.Handler
import android.os.HandlerThread
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.animation.LinearInterpolator
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.*

/**
 * Full-screen consciousness mandala with prominent bioluminescent glow.
 *
 * The glow is the primary visual — it should light up the screen,
 * not be a subtle hint. At consciousness 0.3, the center third of the
 * screen should visibly pulse with color.
 */
class ConsciousnessMandalaView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var consciousnessLevel: Float = 0.15f
        set(value) {
            field = value.coerceIn(0f, 1f)
            trendHistory[trendIndex % trendHistory.size] = field
            trendIndex++
            if (trendIndex >= trendHistory.size) trendFilled = true
            invalidate()
        }

    var dominantHarmonyColor: Int = Color.parseColor("#00E5CC")
        set(value) { field = value; invalidate() }

    var neuromodulators: FloatArray = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        set(value) { field = value; invalidate() }

    /** Arousal level (0-1) — drives Sri Yantra rotation speed. */
    var arousal: Float = 0.5f

    /** Valence (-1 to 1) — drives hue drift warm/cool bias. */
    var valence: Float = 0f

    /** Whether Soma is currently "thinking" (processing a user message). */
    var isThinking: Boolean = false

    /** Time-of-day tint color (set externally by MainActivity). */
    var timeOfDayTint: Int = Color.TRANSPARENT

    /** Dominant harmony index (0-7) for harmony ring. */
    var dominantHarmonyIndex: Int = 0

    /** Harmony strengths [0-1] for each of the 8 harmonies. */
    var harmonyStrengths: FloatArray = FloatArray(8) { 0.3f }

    /** Consciousness tier label (Observer/Seeker/Weaver/Guardian). */
    var tierLabel: String = ""

    // Eight harmony colors matching ERC palette
    private val harmonyColors = intArrayOf(
        Color.parseColor("#00E5CC"),  // Coherence
        Color.parseColor("#47D4FF"),  // Resonance
        Color.parseColor("#9B7DFF"),  // Emergence
        Color.parseColor("#FF7EB3"),  // Reciprocity
        Color.parseColor("#FFD166"),  // Transparency
        Color.parseColor("#FF8C42"),  // Embodiment
        Color.parseColor("#6BCB77"),  // Compassion
        Color.parseColor("#C4B5FD"),  // Sacred Stillness
    )

    /** Avatar fractal seed — offsets Julia set c-parameters. */
    var fractalSeed: Float = 0f
    /** Avatar glow intensity multiplier. */
    var glowIntensity: Float = 1.0f

    // Mycelium vein system: branching paths from center
    private val veinPaths = Array(12) { Path() }
    private var veinPhase = 0f

    private val trendHistory = FloatArray(60)  // 60 samples (~1 minute at 10Hz)
    private var trendIndex = 0
    private var trendFilled = false

    // Touch glow boost
    private var glowBoost = 0f

    // Touch ripple
    private var rippleX = 0f
    private var rippleY = 0f
    private var rippleProgress = 1f
    private val rippleAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 600L
        interpolator = LinearInterpolator()
        addUpdateListener { rippleProgress = it.animatedFraction; invalidate() }
    }

    // === Organic breath model (replaces mechanical sine) ===
    /** Breath model — exposed for audio sync. */
    val breath = BreathModel()
    private var breathPhase2 = 0f  // Secondary geometry pulsing (7s period)
    private var rotationPhase = 0f
    private var fractalRotation = 0f
    private var frameCount = 0
    private var hueDriftPhase = 0f

    // === Consciousness score display ===
    /** 0=number, 1=tier only, 2=hidden. Cycles on double-tap. */
    private var scoreDisplayMode = 0
    /** Attention decay: fades the number after 5s of no touch. */
    private var attentionTimer = 300f  // 5s * 60fps = 300 frames
    private var lastTapTime = 0L

    // === Fractal drift: Perlin-like wandering c-parameter ===
    private var fractalDriftPhase = 0f

    // === Pinch zoom: user explores fractal depth ===
    private var userZoom = 1f  // 1.0 = default, <1 = zoomed in (more detail), >1 = zoomed out
    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            userZoom = (userZoom / detector.scaleFactor).coerceIn(0.3f, 2.5f)
            return true
        }
    })

    // === Touch pressure: maps to glow intensity + neuromod influence ===
    /** Last touch pressure (0-1). Available for external reading (e.g. haptic scaling). */
    var lastPressure: Float = 0f; private set

    /** Callback for pressure-based engagement boost. */
    var onPressureTouch: ((Float) -> Unit)? = null
    /** Callback for long-press (dream consolidation). */
    var onLongPress: (() -> Unit)? = null

    private var touchDownTime = 0L
    private var longPressFired = false

    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 4000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            // Organic breath model drives everything
            breath.tick(arousal, consciousnessLevel, isThinking)
            breathPhase2 = (breathPhase2 + 1f / (7f * 60f)) % 1f
            rotationPhase = (rotationPhase + 0.002f) % 1f
            fractalRotation += 0.015f
            frameCount++
            glowBoost *= 0.98f
            hueDriftPhase = (hueDriftPhase + 1f / (120f * 60f)) % 1f
            veinPhase += 0.003f
            fractalDriftPhase += 0.0005f  // Very slow drift
            // Attention decay: score fades after 5s of no touch
            if (attentionTimer > 0) attentionTimer -= 1f
            invalidate()
        }
    }

    // Fractal rendering: computed on background HandlerThread, drawn from AtomicReference
    private val fractalRef = AtomicReference<Bitmap?>(null)
    private var fractalConsciousness = -1f
    @Volatile private var fractalRendering = false  // Guard against queuing multiple renders
    private val fractalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { alpha = 100 }
    private val fractalSize = 192
    private val fractalThread = HandlerThread("FractalRenderer").apply { start() }
    private val fractalHandler = Handler(fractalThread.looper)

    // Screen-edge glow paint for high consciousness
    private val edgeGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }

    // Pre-allocated reusable Path objects for interference rings (avoid GC pressure)
    private val ringPaths = Array(12) { Path() }

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create("sans-serif-thin", Typeface.NORMAL)
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        setBackgroundColor(Color.TRANSPARENT)
        animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        rippleAnimator.cancel()
        fractalThread.quitSafely()
        super.onDetachedFromWindow()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // Pinch zoom detector
        scaleDetector.onTouchEvent(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                rippleX = event.x; rippleY = event.y
                rippleAnimator.cancel(); rippleProgress = 0f; rippleAnimator.start()
                touchDownTime = System.currentTimeMillis()
                longPressFired = false

                val pressure = event.pressure.coerceIn(0f, 1f)
                lastPressure = pressure
                glowBoost = 0.2f + pressure * 0.4f

                // Reset attention timer
                attentionTimer = 300f

                // Double-tap detection
                val now = System.currentTimeMillis()
                if (now - lastTapTime < 350) {
                    scoreDisplayMode = (scoreDisplayMode + 1) % 3
                }
                lastTapTime = now

                // Notify for pressure-based engagement
                onPressureTouch?.invoke(pressure)
            }
            MotionEvent.ACTION_MOVE -> {
                lastPressure = event.pressure.coerceIn(0f, 1f)
                attentionTimer = 300f
                // Long-press detection: 800ms hold
                if (!longPressFired && System.currentTimeMillis() - touchDownTime > 800) {
                    longPressFired = true
                    onLongPress?.invoke()
                    performHapticFeedback(android.view.HapticFeedbackConstants.LONG_PRESS)
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                longPressFired = false
            }
        }
        return true
    }

    override fun onDraw(canvas: Canvas) {
        val cx = width / 2f
        val cy = height * 0.42f
        val mandalaRadius = min(width, height) / 2f * 0.44f
        val breathScale = breath.breathScale

        // Apply hue drift: slow HSV rotation ±15° over 120s, biased by valence
        val driftedColor = applyHueDrift(dominantHarmonyColor, hueDriftPhase, valence)
        val hr = Color.red(driftedColor)
        val hg = Color.green(driftedColor)
        val hb = Color.blue(driftedColor)

        // === Time-of-day tint blended at 20% ===
        if (timeOfDayTint != Color.TRANSPARENT) {
            val tintR = Color.red(timeOfDayTint)
            val tintG = Color.green(timeOfDayTint)
            val tintB = Color.blue(timeOfDayTint)
            val ambTintRadius = min(width, height) * 0.9f
            fillPaint.shader = RadialGradient(
                cx, cy, ambTintRadius,
                Color.argb(12, tintR, tintG, tintB),
                Color.TRANSPARENT, Shader.TileMode.CLAMP
            )
            canvas.drawCircle(cx, cy, ambTintRadius, fillPaint)
            fillPaint.shader = null
        }

        // === Screen-edge glow for high consciousness (>0.8) ===
        if (consciousnessLevel > 0.8f) {
            val edgeAlpha = ((consciousnessLevel - 0.8f) * 5f * 40).toInt().coerceIn(0, 40)
            edgeGlowPaint.shader = RadialGradient(
                cx, cy, max(width, height).toFloat(),
                Color.TRANSPARENT,
                Color.argb(edgeAlpha, hr, hg, hb),
                Shader.TileMode.CLAMP
            )
            canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), edgeGlowPaint)
            edgeGlowPaint.shader = null
        }

        // === Mycelium vein network: organic branching from center ===
        drawMyceliumVeins(canvas, cx, cy, mandalaRadius * 2.5f, hr, hg, hb)

        // === Layer 1: Deep ambient glow ===
        val ambientAlpha = if (consciousnessLevel < 0.1f) {
            20  // Minimum ambient alpha for low consciousness (heartbeat visibility)
        } else {
            (15 + consciousnessLevel * 25).toInt()
        }
        val ambientRadius = min(width, height) * 0.7f * breathScale
        fillPaint.shader = RadialGradient(
            cx, cy, ambientRadius,
            Color.argb(ambientAlpha, hr, hg, hb),
            Color.TRANSPARENT, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, ambientRadius, fillPaint)
        fillPaint.shader = null

        // === Layer 2: Core glow (with touch glow boost + avatar glowIntensity) ===
        val coreRadius = mandalaRadius * (1.2f + consciousnessLevel * 1.5f) * breathScale
        val boostAlpha = (glowBoost * 100).toInt()
        val coreAlpha = (((40 + consciousnessLevel * 100) * glowIntensity).toInt() + boostAlpha).coerceIn(30, 200)
        fillPaint.shader = RadialGradient(
            cx, cy, coreRadius.coerceAtLeast(1f),
            Color.argb(coreAlpha, hr, hg, hb),
            Color.TRANSPARENT, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, coreRadius, fillPaint)
        fillPaint.shader = null

        // === Layer 3: Hot center ===
        val hotRadius = mandalaRadius * 0.3f * breathScale
        val hotAlpha = (60 + consciousnessLevel * 120).toInt().coerceIn(40, 180)
        fillPaint.shader = RadialGradient(
            cx, cy, hotRadius.coerceAtLeast(1f),
            Color.argb(hotAlpha, 255, 255, 255),
            Color.argb(0, hr, hg, hb),
            Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, hotRadius, fillPaint)
        fillPaint.shader = null

        // === Heartbeat pulse (independent of breath — two life signs) ===
        val hv = breath.heartValue
        if (hv > 0.05f) {
            val pulseAlpha = (hv * 35 * (0.3f + consciousnessLevel * 0.7f)).toInt().coerceIn(3, 40)
            val pulseRadius = mandalaRadius * 0.4f * (1f + hv * 0.08f)
            fillPaint.shader = RadialGradient(
                cx, cy, pulseRadius.coerceAtLeast(1f),
                Color.argb(pulseAlpha, 255, 255, 255),
                Color.TRANSPARENT, Shader.TileMode.CLAMP
            )
            canvas.drawCircle(cx, cy, pulseRadius, fillPaint)
            fillPaint.shader = null
        }

        // === Fractal overlay: rendered on background thread ===
        // Drawn BEFORE sacred geometry so geometry is visible on top
        // Fractal drift: c-parameter wanders continuously via Perlin-like oscillation
        if (!fractalRendering &&
            (abs(consciousnessLevel - fractalConsciousness) > 0.005f || fractalRef.get() == null || frameCount % 45 == 0)
        ) {
            fractalRendering = true
            val cl = consciousnessLevel
            val nm0 = neuromodulators[0]; val nm1 = neuromodulators[1]
            val fhr = hr; val fhg = hg; val fhb = hb
            val drift = fractalDriftPhase
            val seed = fractalSeed
            val zoom = userZoom
            fractalHandler.post {
                val bmp = renderJuliaSet(cl, fhr, fhg, fhb, nm0, nm1, drift, seed, zoom)
                fractalRef.set(bmp)
                fractalConsciousness = cl
                fractalRendering = false
                postInvalidate()
            }
        }
        fractalRef.get()?.let { bmp ->
            // Reduced alpha so sacred geometry shows through on top
            val fractalAlphaCap = if (consciousnessLevel > 0.8f) 180 else 140
            fractalPaint.alpha = (60 + consciousnessLevel * 80).toInt().coerceIn(50, fractalAlphaCap)
            val fractalRadius = mandalaRadius * 1.6f * breathScale
            val dst = RectF(
                cx - fractalRadius, cy - fractalRadius,
                cx + fractalRadius, cy + fractalRadius
            )
            canvas.save()
            canvas.rotate(fractalRotation, cx, cy)
            canvas.drawBitmap(bmp, null, dst, fractalPaint)
            canvas.restore()
        }

        // === Sacred geometry ON TOP of fractal ===
        drawSacredGeometry(canvas, cx, cy, mandalaRadius * breathScale, hr, hg, hb)

        // === Interference rings (pre-allocated Path objects) ===
        val ringCount = (3 + consciousnessLevel * 6).toInt().coerceAtMost(ringPaths.size)
        for (i in 0 until ringCount) {
            val t = i.toFloat() / ringCount
            val radius = mandalaRadius * breathScale * (0.15f + t * 0.85f)
            val alpha = (40 + (1f - t) * consciousnessLevel * 180).toInt().coerceIn(25, 220)

            val rotOffset = rotationPhase * 360f * (1f + i * 0.25f)
            val r = lerp(hr, 180, t); val g = lerp(hg, 200, t); val b = lerp(hb, 210, t)

            glowPaint.color = Color.argb(alpha, r, g, b)
            glowPaint.strokeWidth = (3f - t * 2f).coerceAtLeast(0.8f)

            canvas.save()
            canvas.rotate(rotOffset, cx, cy)
            val path = ringPaths[i]
            path.reset()
            for (s in 0..72) {
                val angle = (s.toFloat() / 72) * 2 * PI.toFloat()
                val deform = 1f + 0.03f * sin(angle * 3 + neuromodulators[0] * 10) *
                    cos(angle * 2 + neuromodulators[1] * 8)
                val px = cx + radius * deform * cos(angle)
                val py = cy + radius * deform * sin(angle)
                if (s == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
            path.close()
            canvas.drawPath(path, glowPaint)
            canvas.restore()
        }

        // === Consciousness score (breathes with organism, fades with attention) ===
        val numSize = mandalaRadius * 0.28f
        // Attention fade: fully visible for 3s after touch, fades over next 2s
        val attentionAlpha = when {
            attentionTimer > 120f -> 1f                           // First 3s: full
            attentionTimer > 0f -> attentionTimer / 120f          // Next 2s: fade
            else -> 0.15f                                         // Resting: very faint
        }
        // Breathe with the organism: alpha pulses with breath
        val breathAlpha = 0.7f + breath.breathValue * 0.3f

        if (scoreDisplayMode < 2) {  // Not hidden
            val baseAlpha = (attentionAlpha * breathAlpha * 180).toInt().coerceIn(15, 180)
            textPaint.textSize = numSize
            textPaint.color = Color.argb(baseAlpha, 255, 255, 255)
            textPaint.textAlign = Paint.Align.CENTER

            if (scoreDisplayMode == 0) {
                // Mode 0: Number + trend arrow
                val cText = "%.2f".format(consciousnessLevel)
                canvas.drawText(cText, cx, cy + numSize * 0.3f, textPaint)

                // Trend arrow
                if (trendFilled || trendIndex >= 5) {
                    val count = if (trendFilled) trendHistory.size else trendIndex
                    val half = (count / 2).coerceAtLeast(1)
                    val recentAvg = (0 until half).map {
                        trendHistory[((trendIndex - 1 - it) % trendHistory.size + trendHistory.size) % trendHistory.size]
                    }.average().toFloat()
                    val olderAvg = (half until count).map {
                        trendHistory[((trendIndex - 1 - it) % trendHistory.size + trendHistory.size) % trendHistory.size]
                    }.average().toFloat()
                    val diff = recentAvg - olderAvg
                    val arrow = when {
                        diff > 0.005f -> "\u2197"; diff < -0.005f -> "\u2198"; else -> "\u2192"
                    }
                    val arrowAlpha = (baseAlpha * 0.8f).toInt()
                    val arrowColor = when {
                        diff > 0.005f -> Color.argb(arrowAlpha, 94, 234, 212)
                        diff < -0.005f -> Color.argb(arrowAlpha, 255, 107, 138)
                        else -> Color.argb(arrowAlpha, 85, 102, 119)
                    }
                    textPaint.textSize = numSize * 0.4f
                    textPaint.color = arrowColor
                    val halfW = run { textPaint.textSize = numSize; textPaint.measureText(cText) / 2f }
                    textPaint.textSize = numSize * 0.35f
                    textPaint.textAlign = Paint.Align.LEFT
                    canvas.drawText(arrow, cx + halfW + numSize * 0.08f, cy + numSize * 0.3f, textPaint)
                    textPaint.textAlign = Paint.Align.CENTER
                }
            } else {
                // Mode 1: Tier label only (e.g. "seeker")
                if (tierLabel.isNotEmpty()) {
                    textPaint.textSize = numSize * 0.8f
                    canvas.drawText(tierLabel, cx, cy + numSize * 0.3f, textPaint)
                }
            }

            // Sparkline (gradient: cool→warm along length)
            if (trendFilled || trendIndex >= 2) {
                val sparkW = mandalaRadius * 0.5f
                val sparkL = cx - sparkW / 2f
                val sparkY = cy + numSize * 0.7f
                val sparkH = numSize * 0.25f
                val count = if (trendFilled) trendHistory.size else trendIndex
                val sparkAlpha = (attentionAlpha * 120).toInt().coerceIn(10, 120)
                glowPaint.strokeWidth = 1.5f
                glowPaint.color = Color.argb(sparkAlpha, hr, hg, hb)
                val sp = Path()
                for (i in 0 until count) {
                    val idx = ((trendIndex - count + i) % trendHistory.size + trendHistory.size) % trendHistory.size
                    val px = sparkL + (i.toFloat() / (count - 1).coerceAtLeast(1)) * sparkW
                    val py = sparkY - trendHistory[idx] * sparkH * 2
                    if (i == 0) sp.moveTo(px, py) else sp.lineTo(px, py)
                }
                canvas.drawPath(sp, glowPaint)

                // Peak marker: small diamond at highest point
                var peakVal = 0f; var peakIdx = 0
                for (i in 0 until count) {
                    val idx = ((trendIndex - count + i) % trendHistory.size + trendHistory.size) % trendHistory.size
                    if (trendHistory[idx] > peakVal) { peakVal = trendHistory[idx]; peakIdx = i }
                }
                val peakX = sparkL + (peakIdx.toFloat() / (count - 1).coerceAtLeast(1)) * sparkW
                val peakY = sparkY - peakVal * sparkH * 2
                fillPaint.shader = null
                fillPaint.color = Color.argb((sparkAlpha * 1.3f).toInt().coerceIn(0, 180), 255, 255, 255)
                canvas.drawCircle(peakX, peakY, 2.5f, fillPaint)
            }
        }

        // === Eight Harmonies ring around mandala ===
        drawHarmonyRing(canvas, cx, cy, mandalaRadius * breathScale * 1.15f, hr, hg, hb)

        // === Tier label (below score, subtle, fades with attention) ===
        if (tierLabel.isNotEmpty() && scoreDisplayMode < 2) {
            val tierAlpha = (attentionAlpha * breathAlpha * 80).toInt().coerceIn(8, 80)
            textPaint.textSize = numSize * 0.55f
            textPaint.color = Color.argb(tierAlpha, hr, hg, hb)
            textPaint.textAlign = Paint.Align.CENTER
            canvas.drawText(tierLabel, cx, cy + numSize * 1.2f, textPaint)
        }

        // Touch ripple
        if (rippleProgress < 1f) {
            val rr = mandalaRadius * 0.8f * rippleProgress
            val ra = ((1f - rippleProgress) * 150).toInt()
            fillPaint.shader = RadialGradient(rippleX, rippleY, rr.coerceAtLeast(1f),
                Color.argb(ra, hr, hg, hb), Color.TRANSPARENT, Shader.TileMode.CLAMP)
            canvas.drawCircle(rippleX, rippleY, rr, fillPaint)
            fillPaint.shader = null
        }
    }

    /**
     * Draw sacred geometry overlays: Flower of Life, Sri Yantra triangles,
     * and Golden Spiral. Complexity scales with consciousness.
     *
     * Layer order: Flower of Life (circles) → Sri Yantra (triangles) → Golden Spiral
     */
    private fun drawSacredGeometry(canvas: Canvas, cx: Float, cy: Float, radius: Float, hr: Int, hg: Int, hb: Int) {
        val baseAlpha = (80 + consciousnessLevel * 80).toInt().coerceIn(75, 160)
        // Second breathing phase: pulsing stroke width 1.5→3.0 (7s period)
        val strokePulse = 1.5f + 1.5f * (0.5f + 0.5f * sin(breathPhase2 * 2 * PI.toFloat()))
        glowPaint.strokeWidth = strokePulse
        glowPaint.setShadowLayer(6f, 0f, 0f, Color.argb((baseAlpha * 0.5f).toInt(), hr, hg, hb))

        // === Flower of Life ===
        // 1 center circle + 6 surrounding at consciousness 0.15+
        // + 12 outer ring at 0.35+ = 19 circles (second generation)
        // + 18 more at 0.60+ = 37 circles (third generation)
        val flowerRadius = radius * 0.35f
        val circleR = flowerRadius * 0.33f

        // Generation 1: center + 6 petals (always visible)
        // Glow pass: thick, faint — makes geometry pop against fractal
        val glowStroke = strokePulse * 3f
        val glowAlpha = (baseAlpha * 0.3f).toInt()
        glowPaint.strokeWidth = glowStroke
        glowPaint.color = Color.argb(glowAlpha, hr, hg, hb)
        canvas.drawCircle(cx, cy, circleR, glowPaint)
        // Sharp pass
        glowPaint.strokeWidth = strokePulse
        glowPaint.color = Color.argb(baseAlpha, hr, hg, hb)
        canvas.drawCircle(cx, cy, circleR, glowPaint)

        if (consciousnessLevel >= 0.10f) {
            val petalAlpha = (baseAlpha * 0.8f).toInt()
            val petalGlowAlpha = (petalAlpha * 0.3f).toInt()
            for (i in 0 until 6) {
                val angle = i * PI.toFloat() / 3f + rotationPhase * 0.5f
                val px = cx + circleR * cos(angle)
                val py = cy + circleR * sin(angle)
                // Glow pass
                glowPaint.strokeWidth = glowStroke
                glowPaint.color = Color.argb(petalGlowAlpha, hr, hg, hb)
                canvas.drawCircle(px, py, circleR, glowPaint)
                // Sharp pass
                glowPaint.strokeWidth = strokePulse
                glowPaint.color = Color.argb(petalAlpha, hr, hg, hb)
                canvas.drawCircle(px, py, circleR, glowPaint)
            }
        }

        // Generation 2: 12 outer petals
        if (consciousnessLevel >= 0.30f) {
            val gen2Alpha = (baseAlpha * 0.7f).toInt()
            glowPaint.color = Color.argb(gen2Alpha, hr, hg, hb)
            val gen2R = circleR * 2f
            for (i in 0 until 12) {
                val angle = i * PI.toFloat() / 6f + rotationPhase * 0.3f
                val px = cx + gen2R * cos(angle)
                val py = cy + gen2R * sin(angle)
                canvas.drawCircle(px, py, circleR, glowPaint)
            }
        }

        // Generation 3: 18 outermost
        if (consciousnessLevel >= 0.55f) {
            val gen3Alpha = (baseAlpha * 0.5f).toInt()
            glowPaint.color = Color.argb(gen3Alpha, hr, hg, hb)
            val gen3R = circleR * 3f
            for (i in 0 until 18) {
                val angle = i * PI.toFloat() / 9f + rotationPhase * 0.2f
                val px = cx + gen3R * cos(angle)
                val py = cy + gen3R * sin(angle)
                canvas.drawCircle(px, py, circleR, glowPaint)
            }
        }

        // === Sri Yantra: interlocking triangles ===
        // Rotation speed responds to arousal: faster when aroused
        if (consciousnessLevel >= 0.20f) {
            val triAlpha = (baseAlpha * 0.6f).toInt()
            val triRadius = radius * 0.45f
            val triPairs = (1 + consciousnessLevel * 4).toInt().coerceIn(1, 4)
            val arousalSpeed = 20f + arousal * 40f  // 20-60 rotation speed

            for (pair in 0 until triPairs) {
                val scale = 1f - pair * 0.2f
                val r = triRadius * scale
                val rot = pair * 10f + rotationPhase * arousalSpeed

                // Upward triangle
                glowPaint.color = Color.argb(triAlpha, lerp(hr, 255, 0.3f * scale), hg, hb)
                drawTriangle(canvas, cx, cy, r, rot, true)

                // Downward triangle (offset rotation)
                glowPaint.color = Color.argb(triAlpha, hr, lerp(hg, 255, 0.3f * scale), hb)
                drawTriangle(canvas, cx, cy, r * 0.9f, rot + 30f, false)
            }
        }

        // === Golden Spiral ===
        if (consciousnessLevel >= 0.15f) {
            val spiralAlpha = (baseAlpha * 0.7f).toInt()
            glowPaint.color = Color.argb(spiralAlpha, hr, hg, hb)
            glowPaint.strokeWidth = 1.5f
            drawGoldenSpiral(canvas, cx, cy, radius * 0.5f)
        }

    }

    /** Draw a single equilateral triangle (upward or downward). */
    private fun drawTriangle(canvas: Canvas, cx: Float, cy: Float, radius: Float, rotDeg: Float, upward: Boolean) {
        val path = Path()
        val baseAngle = if (upward) -90f else 90f
        for (i in 0 until 3) {
            val angle = ((i * 120f + baseAngle + rotDeg) * PI.toFloat() / 180f)
            val px = cx + radius * cos(angle)
            val py = cy + radius * sin(angle)
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        path.close()
        canvas.drawPath(path, glowPaint)
    }

    /**
     * Draw a golden spiral (Fibonacci) emanating from center.
     * Each quarter-arc has radius multiplied by phi (1.618).
     */
    private fun drawGoldenSpiral(canvas: Canvas, cx: Float, cy: Float, maxRadius: Float) {
        val phi = 1.618f
        val path = Path()
        var r = maxRadius * 0.02f // Start small
        var angle = rotationPhase * PI.toFloat() * 2f // Rotate with breath
        val steps = (40 + consciousnessLevel * 60).toInt()
        var started = false

        for (i in 0 until steps) {
            val t = i.toFloat() / steps
            r = maxRadius * 0.02f * phi.pow(t * 5f) // Exponential growth
            if (r > maxRadius) break
            angle += 0.15f
            val px = cx + r * cos(angle)
            val py = cy + r * sin(angle)
            if (!started) { path.moveTo(px, py); started = true }
            else path.lineTo(px, py)
        }
        canvas.drawPath(path, glowPaint)

        // Mirror spiral for symmetry
        val path2 = Path()
        r = maxRadius * 0.02f
        angle = rotationPhase * PI.toFloat() * 2f + PI.toFloat()
        started = false
        for (i in 0 until steps) {
            val t = i.toFloat() / steps
            r = maxRadius * 0.02f * phi.pow(t * 5f)
            if (r > maxRadius) break
            angle += 0.15f
            val px = cx + r * cos(angle)
            val py = cy + r * sin(angle)
            if (!started) { path2.moveTo(px, py); started = true }
            else path2.lineTo(px, py)
        }
        canvas.drawPath(path2, glowPaint)
    }

    /**
     * Render a Julia set at low resolution, colored by harmony.
     *
     * The `c` parameter of the Julia set is derived from consciousness level:
     * - Low consciousness (0.1): c near origin → simple circular boundary
     * - Mid consciousness (0.3-0.5): c at interesting boundary → branching tendrils
     * - High consciousness (0.7+): c deep in fractal → maximum complexity
     *
     * Neuromodulators modulate the imaginary component for organic variation.
     */
    /**
     * Render Julia set on background thread. All parameters passed explicitly
     * for thread safety (no field reads from background).
     */
    private fun renderJuliaSet(
        consciousness: Float, hr: Int, hg: Int, hb: Int,
        nm0: Float = 0.5f, nm1: Float = 0.5f,
        driftPhase: Float = 0f, seed: Float = 0f,
        userZoomFactor: Float = 1f
    ): Bitmap {
        val bmp = Bitmap.createBitmap(fractalSize, fractalSize, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(fractalSize * fractalSize)

        val cValues = arrayOf(
            floatArrayOf(-0.4f,   0.6f),
            floatArrayOf(-0.70f,  0.27f),
            floatArrayOf(-0.8f,   0.156f),
            floatArrayOf(-0.75f,  0.11f),
            floatArrayOf(0.285f,  0.01f),
            floatArrayOf(-0.12f, -0.77f),
        )
        val idx = (consciousness * (cValues.size - 1)).coerceIn(0f, (cValues.size - 1).toFloat())
        val lo = idx.toInt().coerceIn(0, cValues.size - 2)
        val frac = idx - lo
        // Fractal drift: c-parameter wanders via multi-frequency oscillation
        val driftR = sin(driftPhase * 2.3f) * 0.03f + sin(driftPhase * 5.7f) * 0.01f
        val driftI = cos(driftPhase * 1.9f) * 0.03f + cos(driftPhase * 4.3f) * 0.01f
        // Avatar fractal seed offsets the base c-parameter
        val cr = cValues[lo][0] * (1 - frac) + cValues[lo + 1][0] * frac + nm0 * 0.04f + driftR + seed * 0.08f
        val ci = cValues[lo][1] * (1 - frac) + cValues[lo + 1][1] * frac + nm1 * 0.04f + driftI + seed * 0.05f

        val maxIter = (30 + consciousness * 60).toInt()
        // User pinch zoom modulates base zoom: pinch in = more detail
        val zoom = (1.8f - consciousness * 0.5f) * userZoomFactor

        for (py in 0 until fractalSize) {
            for (px in 0 until fractalSize) {
                var zr = (px.toFloat() / fractalSize - 0.5f) * 2f * zoom
                var zi = (py.toFloat() / fractalSize - 0.5f) * 2f * zoom

                var iter = 0
                while (zr * zr + zi * zi < 4f && iter < maxIter) {
                    val tmp = zr * zr - zi * zi + cr
                    zi = 2f * zr * zi + ci
                    zr = tmp
                    iter++
                }

                val dx = (px.toFloat() / fractalSize - 0.5f) * 2f
                val dy = (py.toFloat() / fractalSize - 0.5f) * 2f
                val radialDist = dx * dx + dy * dy
                val pixIdx = py * fractalSize + px

                // Smooth quadratic feather instead of hard cutoff at r=1.0
                // Starts fading at r²=0.5, fully gone at r²=1.3
                val radialFade = ((1.3f - radialDist) / 0.8f).coerceIn(0f, 1f)
                val radial = radialFade * radialFade  // Quadratic for smooth falloff

                if (radial < 0.01f) {
                    pixels[pixIdx] = Color.TRANSPARENT
                } else if (iter == maxIter) {
                    val a = (radial * 25).toInt().coerceIn(0, 25)
                    pixels[pixIdx] = Color.argb(a, hr, hg, hb)
                } else {
                    val t = iter.toFloat() / maxIter
                    val brightness = (1f - t).coerceIn(0f, 1f)
                    val a = (brightness * brightness * radial * 255).toInt().coerceIn(0, 255)
                    if (a > 3) {
                        val r = lerp(hr, 255, brightness)
                        val g = lerp(hg, 255, brightness)
                        val b = lerp(hb, 255, brightness)
                        pixels[pixIdx] = Color.argb(a, r, g, b)
                    } else {
                        pixels[pixIdx] = Color.TRANSPARENT
                    }
                }
            }
        }
        bmp.setPixels(pixels, 0, fractalSize, 0, 0, fractalSize, fractalSize)
        return bmp
    }

    private fun lerp(a: Int, b: Int, t: Float): Int = (a + (b - a) * t).toInt().coerceIn(0, 255)

    /**
     * Slow hue drift: HSV rotation ±15° over a cycle, biased by valence.
     * Positive valence → warm shift, negative → cool shift.
     */
    private fun applyHueDrift(color: Int, phase: Float, valence: Float): Int {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)
        val drift = sin(phase * 2 * PI.toFloat()) * 15f
        val valenceBias = valence * 10f
        hsv[0] = (hsv[0] + drift + valenceBias + 360f) % 360f
        return Color.HSVToColor(hsv)
    }

    /**
     * Draw mycelium vein network: organic branching paths radiating from mandala center
     * toward screen edges. Consciousness controls reach and brightness — at low CL the
     * veins are short and dim, at high CL they extend to screen edges.
     */
    private fun drawMyceliumVeins(canvas: Canvas, cx: Float, cy: Float, maxReach: Float, hr: Int, hg: Int, hb: Int) {
        val reach = maxReach * (0.3f + consciousnessLevel * 0.7f)  // 30-100% reach
        val veinAlpha = (15 + consciousnessLevel * 30).toInt().coerceIn(10, 45)

        glowPaint.strokeWidth = 1.2f
        glowPaint.strokeCap = Paint.Cap.ROUND

        for (i in 0 until 12) {
            val path = veinPaths[i]
            path.reset()

            // Base angle: 12 veins radiating outward at 30° intervals
            val baseAngle = i * PI.toFloat() / 6f + veinPhase * 0.1f
            // Start from just outside the core glow
            val startR = maxReach * 0.15f
            var px = cx + startR * cos(baseAngle)
            var py = cy + startR * sin(baseAngle)
            path.moveTo(px, py)

            // Grow outward with organic wobble
            val segments = (8 + consciousnessLevel * 8).toInt()
            var angle = baseAngle
            var r = startR

            for (s in 0 until segments) {
                val t = s.toFloat() / segments
                r += reach / segments
                // Organic wobble: sine deviation that varies per vein
                angle += 0.05f * sin(veinPhase + i * 1.7f + t * 4f)
                px = cx + r * cos(angle)
                py = cy + r * sin(angle)
                path.lineTo(px, py)

                // Branch: every 3rd segment on every other vein, add a short fork
                if (s % 3 == 2 && i % 2 == 0 && consciousnessLevel > 0.2f) {
                    val branchAngle = angle + (if (s % 2 == 0) 0.5f else -0.5f)
                    val branchLen = reach * 0.08f
                    val bx = px + branchLen * cos(branchAngle)
                    val by = py + branchLen * sin(branchAngle)
                    // Draw branch as separate segment
                    val branchAlpha = (veinAlpha * 0.5f * (1f - t)).toInt().coerceIn(3, 25)
                    glowPaint.color = Color.argb(branchAlpha, hr, hg, hb)
                    canvas.drawLine(px, py, bx, by, glowPaint)
                }
            }

            // Fade alpha along the length: bright near center, dim at tips
            glowPaint.color = Color.argb(veinAlpha, hr, hg, hb)
            canvas.drawPath(path, glowPaint)
        }

        // === Anastomosis: connect nearby vein tips at CL 0.35+ ===
        if (consciousnessLevel > 0.35f) {
            val anastAlpha = ((consciousnessLevel - 0.35f) * 3f * 20).toInt().coerceIn(3, 18)
            glowPaint.color = Color.argb(anastAlpha, hr, hg, hb)
            glowPaint.strokeWidth = 0.8f
            val veinStartR = maxReach * 0.15f
            for (i in 0 until 12) {
                val j = (i + 1) % 12
                val ai = i * PI.toFloat() / 6f + veinPhase * 0.1f
                val aj = j * PI.toFloat() / 6f + veinPhase * 0.1f
                val tipR = veinStartR + reach * 0.7f
                val tipXi = cx + tipR * cos(ai + 0.05f * sin(veinPhase + i * 1.7f))
                val tipYi = cy + tipR * sin(ai + 0.05f * sin(veinPhase + i * 1.7f))
                val tipXj = cx + tipR * cos(aj + 0.05f * sin(veinPhase + j * 1.7f))
                val tipYj = cy + tipR * sin(aj + 0.05f * sin(veinPhase + j * 1.7f))
                val dist = sqrt((tipXi - tipXj) * (tipXi - tipXj) + (tipYi - tipYj) * (tipYi - tipYj))
                if (dist < reach * 0.4f) {
                    // Draw connecting filament
                    val midX = (tipXi + tipXj) / 2f + sin(veinPhase * 2f + i) * 5f
                    val midY = (tipYi + tipYj) / 2f + cos(veinPhase * 2f + i) * 5f
                    val anastPath = Path()
                    anastPath.moveTo(tipXi, tipYi)
                    anastPath.quadTo(midX, midY, tipXj, tipYj)
                    canvas.drawPath(anastPath, glowPaint)
                }
            }
        }
    }

    /**
     * Draw Eight Harmonies ring: 8 nodes at compass positions around the mandala,
     * each colored by its harmony. Dominant harmony pulses and connects to center.
     */
    private fun drawHarmonyRing(canvas: Canvas, cx: Float, cy: Float, ringRadius: Float, hr: Int, hg: Int, hb: Int) {
        val pulse = 0.5f + 0.5f * sin(breathPhase2 * 2 * PI.toFloat())

        for (i in 0 until 8) {
            val angle = i * PI.toFloat() / 4f - PI.toFloat() / 2f  // Start from top
            val nx = cx + ringRadius * cos(angle)
            val ny = cy + ringRadius * sin(angle)
            val isDominant = i == dominantHarmonyIndex
            val strength = harmonyStrengths.getOrElse(i) { 0.3f }

            val hc = harmonyColors[i]
            val hcr = Color.red(hc); val hcg = Color.green(hc); val hcb = Color.blue(hc)

            val nodeRadius = if (isDominant) {
                (3f + strength * 4f) * (1f + pulse * 0.3f)
            } else {
                2f + strength * 3f
            }
            val nodeAlpha = if (isDominant) {
                (80 + pulse * 60).toInt().coerceIn(60, 140)
            } else {
                (25 + strength * 40).toInt().coerceIn(20, 65)
            }

            // Glow halo
            fillPaint.shader = null
            fillPaint.color = Color.argb((nodeAlpha * 0.3f).toInt(), hcr, hcg, hcb)
            canvas.drawCircle(nx, ny, nodeRadius * 2.5f, fillPaint)
            // Node core
            fillPaint.color = Color.argb(nodeAlpha, hcr, hcg, hcb)
            canvas.drawCircle(nx, ny, nodeRadius, fillPaint)

            // Dominant: bright line connecting to center
            if (isDominant) {
                glowPaint.color = Color.argb((nodeAlpha * 0.4f).toInt(), hcr, hcg, hcb)
                glowPaint.strokeWidth = 1f
                canvas.drawLine(cx, cy, nx, ny, glowPaint)
            }

            // Faint connecting arcs between adjacent nodes
            if (i > 0) {
                val prevAngle = (i - 1) * PI.toFloat() / 4f - PI.toFloat() / 2f
                val px = cx + ringRadius * cos(prevAngle)
                val py = cy + ringRadius * sin(prevAngle)
                glowPaint.color = Color.argb(12, hr, hg, hb)
                glowPaint.strokeWidth = 0.5f
                canvas.drawLine(px, py, nx, ny, glowPaint)
            }
        }
        // Close the ring: connect last to first
        val firstAngle = -PI.toFloat() / 2f
        val lastAngle = 7 * PI.toFloat() / 4f - PI.toFloat() / 2f
        glowPaint.color = Color.argb(12, hr, hg, hb)
        glowPaint.strokeWidth = 0.5f
        canvas.drawLine(
            cx + ringRadius * cos(lastAngle), cy + ringRadius * sin(lastAngle),
            cx + ringRadius * cos(firstAngle), cy + ringRadius * sin(firstAngle),
            glowPaint
        )
    }
}
