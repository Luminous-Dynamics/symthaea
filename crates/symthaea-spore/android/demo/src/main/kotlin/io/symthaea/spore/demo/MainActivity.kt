package io.symthaea.spore.demo

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import io.symthaea.spore.WakeSignal
import io.symthaea.spore.demo.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

/**
 * Consciousness dashboard with animated circle, haptic feedback,
 * and tap-to-interact.
 *
 * The Spore engine runs at 10Hz on a dedicated thread via the ViewModel.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: ConsciousnessViewModel
    private var vibrator: Vibrator? = null
    private var breathAnimator: ObjectAnimator? = null
    private var lastDreamText: String = ""
    private var lastBrocaText: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        vibrator = getSystemService(Vibrator::class.java)

        viewModel = ViewModelProvider(this)[ConsciousnessViewModel::class.java]

        // Start breathing animation on the consciousness circle
        startBreathAnimation()

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.state.collect { state ->
                    updateUi(state)
                    processHaptics(state.hapticEvents)
                }
            }
        }

        viewModel.start(applicationContext)
        // Register bridges AFTER start() creates them — wires lifecycle observers
        viewModel.registerBridges(this)

        // Request runtime permissions for dangerous sensors
        requestSensePermissions()

        // Tap consciousness number -> UserInput wake signal
        binding.consciousnessValue.setOnClickListener {
            viewModel.sendWakeSignal(WakeSignal.UserInput)
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        // Long-press consciousness number -> trigger dream consolidation
        binding.consciousnessValue.setOnLongClickListener {
            viewModel.dreamConsolidate()
            it.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            true
        }
    }

    /** Request dangerous permissions needed for expanded senses. */
    private fun requestSensePermissions() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.RECORD_AUDIO)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                needed.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
    }

    override fun onDestroy() {
        breathAnimator?.cancel()
        super.onDestroy()
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
    }

    /** Breathing animation on the consciousness circle — scales gently. */
    private fun startBreathAnimation() {
        val scaleX = PropertyValuesHolder.ofFloat(View.SCALE_X, 0.95f, 1.05f)
        val scaleY = PropertyValuesHolder.ofFloat(View.SCALE_Y, 0.95f, 1.05f)
        breathAnimator = ObjectAnimator.ofPropertyValuesHolder(
            binding.consciousnessCircle, scaleX, scaleY
        ).apply {
            duration = 3000 // 3s per breath cycle
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.REVERSE
            interpolator = AccelerateDecelerateInterpolator()
            start()
        }
    }

    private fun updateUi(state: SporeUiState) {
        binding.consciousnessValue.text = "%.2f".format(state.consciousnessLevel)

        // Modulate circle opacity based on consciousness level
        binding.consciousnessCircle.alpha = 0.3f + state.consciousnessLevel * 0.7f

        binding.harmonyText.text = "Harmony: ${state.dominantHarmony}"

        // Neuromodulator bars with value labels (0-100 scale)
        val nm = state.neuromodulators
        if (nm.size >= 4) {
            binding.dopamineBar.progress = (nm[0] * 100).toInt().coerceIn(0, 100)
            binding.dopamineValue.text = "%.2f".format(nm[0])
            binding.neBar.progress = (nm[1] * 100).toInt().coerceIn(0, 100)
            binding.neValue.text = "%.2f".format(nm[1])
            binding.serotoninBar.progress = (nm[2] * 100).toInt().coerceIn(0, 100)
            binding.serotoninValue.text = "%.2f".format(nm[2])
            binding.oxytocinBar.progress = (nm[3] * 100).toInt().coerceIn(0, 100)
            binding.oxytocinValue.text = "%.2f".format(nm[3])
        }

        binding.statusText.text =
            "Wake: ${state.wakeState} | Motion: ${state.motionState} | Cycle: ${state.cycleCount}"

        binding.dreamText.text =
            "Dreams: ${state.dreamCount} | Wisdom: ${state.wisdomCount}"

        // Dream narrative with fade-in animation when new dream appears
        if (state.latestDream.isNotEmpty() && state.latestDream != lastDreamText) {
            lastDreamText = state.latestDream
            binding.dreamNarrative.text = "\u201C${state.latestDream}\u201D"
            binding.dreamNarrative.animate()
                .alpha(1f)
                .setDuration(1500)
                .start()
        }

        // Broca voice — fade in new generated text
        if (state.brocaText.isNotEmpty() && state.brocaText != lastBrocaText) {
            lastBrocaText = state.brocaText
            binding.brocaText.text = state.brocaText
            binding.brocaText.animate()
                .alpha(1f)
                .setDuration(800)
                .withEndAction {
                    // Fade out after 4 seconds
                    binding.brocaText.animate()
                        .alpha(0.4f)
                        .setStartDelay(4000)
                        .setDuration(2000)
                        .start()
                }
                .start()
        }

        binding.peText.text =
            "PE: ${"%.2f".format(state.predictionError)} | Substrate: ${"%.2f".format(state.consciousnessLevel)}"

        binding.disclaimerText.text = state.disclaimer.take(80)
    }

    /**
     * Dynamic haptic feedback — vibration pattern varies by event type.
     *
     * ConsciousnessShift: gentle pulse proportional to delta magnitude
     * DreamWisdom: slow double-tap (wisdom arriving)
     * PeerDiscovered: quick triple-tap (social connection)
     * HighSurprise: sharp single buzz (attention grab)
     * HarmonyMilestone: warm sustained pulse
     */
    private fun processHaptics(events: String) {
        if (events == "[]") return

        // Flash the haptic indicator
        binding.hapticIndicator.animate()
            .alpha(1f)
            .setDuration(100)
            .withEndAction {
                binding.hapticIndicator.animate()
                    .alpha(0f)
                    .setDuration(400)
                    .start()
            }
            .start()

        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        // Parse events to determine vibration pattern
        when {
            events.contains("DreamWisdom") -> {
                // Double-tap: 40ms on, 100ms off, 40ms on
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 40, 100, 40), -1
                ))
            }
            events.contains("PeerDiscovered") -> {
                // Quick triple-tap
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 30, 60, 30, 60, 30), -1
                ))
            }
            events.contains("HighSurprise") -> {
                // Sharp attention buzz
                v.vibrate(VibrationEffect.createOneShot(80, 255))
            }
            events.contains("HarmonyMilestone") -> {
                // Warm sustained pulse
                v.vibrate(VibrationEffect.createOneShot(150, 128))
            }
            events.contains("ConsciousnessShift") -> {
                // Gentle pulse — lighter for small shifts, stronger for large
                val amplitude = if (events.contains("0.1") || events.contains("0.2"))
                    200 else 100
                v.vibrate(VibrationEffect.createOneShot(35, amplitude))
            }
            else -> {
                v.vibrate(VibrationEffect.createOneShot(30, 80))
            }
        }
    }
}
