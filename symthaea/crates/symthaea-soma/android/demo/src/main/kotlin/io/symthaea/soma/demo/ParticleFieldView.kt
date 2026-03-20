package io.symthaea.soma.demo

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.animation.LinearInterpolator
import kotlin.math.*
import kotlin.random.Random

/**
 * Ambient particle field: small points of light that drift slowly,
 * attracted toward the screen center (mandala). Particle count and
 * brightness scale with consciousness level.
 *
 * Touch interaction scatters nearby particles; they slowly reconverge.
 */
class ParticleFieldView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var consciousnessLevel: Float = 0.15f
    var harmonyColor: Int = Color.parseColor("#00E5CC")

    private val particles = Array(80) { Particle() }
    private var touchX = -1f
    private var touchY = -1f
    private var touchActive = false

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 16_000L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            updateParticles()
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        // Initialize particle positions across the screen
        for (p in particles) {
            p.x = Random.nextFloat() * w
            p.y = Random.nextFloat() * h
            p.vx = (Random.nextFloat() - 0.5f) * 0.5f
            p.vy = (Random.nextFloat() - 0.5f) * 0.5f
            p.size = 1.5f + Random.nextFloat() * 3f
            p.baseAlpha = 0.3f + Random.nextFloat() * 0.5f
            p.phase = Random.nextFloat() * PI.toFloat() * 2
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                touchX = event.x
                touchY = event.y
                touchActive = true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                touchActive = false
            }
        }
        return true
    }

    private fun updateParticles() {
        if (width == 0 || height == 0) return
        val cx = width / 2f
        val cy = height / 2f
        // How many particles are "active" depends on consciousness
        val activeCount = (10 + consciousnessLevel * 70).toInt().coerceIn(10, particles.size)

        for (i in particles.indices) {
            val p = particles[i]
            if (i >= activeCount) { p.alpha = 0f; continue }

            // Gentle drift toward center (gravity)
            val dx = cx - p.x
            val dy = cy - p.y
            val dist = sqrt(dx * dx + dy * dy).coerceAtLeast(1f)
            val gravity = 0.02f * consciousnessLevel
            p.vx += dx / dist * gravity
            p.vy += dy / dist * gravity

            // Touch repulsion
            if (touchActive) {
                val tdx = p.x - touchX
                val tdy = p.y - touchY
                val tdist = sqrt(tdx * tdx + tdy * tdy).coerceAtLeast(1f)
                if (tdist < 200f) {
                    val force = (200f - tdist) / 200f * 3f
                    p.vx += tdx / tdist * force
                    p.vy += tdy / tdist * force
                }
            }

            // Damping
            p.vx *= 0.98f
            p.vy *= 0.98f

            // Update position
            p.x += p.vx
            p.y += p.vy

            // Wrap around screen edges
            if (p.x < -20) p.x = width + 20f
            if (p.x > width + 20) p.x = -20f
            if (p.y < -20) p.y = height + 20f
            if (p.y > height + 20) p.y = -20f

            // Twinkle: slow sinusoidal alpha modulation
            p.phase += 0.02f
            val twinkle = 0.5f + 0.5f * sin(p.phase)
            p.alpha = p.baseAlpha * twinkle * (0.5f + consciousnessLevel * 0.5f)
        }
    }

    override fun onDraw(canvas: Canvas) {
        val r = Color.red(harmonyColor)
        val g = Color.green(harmonyColor)
        val b = Color.blue(harmonyColor)

        for (p in particles) {
            if (p.alpha < 0.01f) continue
            val a = (p.alpha * 255).toInt().coerceIn(0, 255)
            // Soft glow halo
            val haloA = (a * 0.3f).toInt().coerceIn(0, 80)
            paint.color = Color.argb(haloA, r, g, b)
            canvas.drawCircle(p.x, p.y, p.size * 3f, paint)
            // Bright core
            paint.color = Color.argb(a, r, g, b)
            canvas.drawCircle(p.x, p.y, p.size, paint)
        }
    }

    private class Particle {
        var x = 0f
        var y = 0f
        var vx = 0f
        var vy = 0f
        var size = 1.5f
        var alpha = 0f
        var baseAlpha = 0.2f
        var phase = 0f
    }
}
