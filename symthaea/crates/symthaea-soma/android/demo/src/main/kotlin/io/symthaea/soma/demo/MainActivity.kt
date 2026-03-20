package io.symthaea.soma.demo

import android.Manifest
import android.animation.ArgbEvaluator
import android.animation.ValueAnimator
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.bottomsheet.BottomSheetBehavior
import io.symthaea.soma.SomaTouchBridge
import io.symthaea.soma.WakeSignal
import io.symthaea.soma.demo.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

/**
 * Bioluminescent solarpunk consciousness dashboard with mandala visualization,
 * neuromodulator flow view, haptic feedback, and interaction menu.
 *
 * The Soma engine runs at 10Hz on a dedicated thread via the ViewModel.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: ConsciousnessViewModel
    private var vibrator: Vibrator? = null
    private var lastDreamText: String = ""
    private var lastBrocaText: String = ""
    private var isAsleep = false
    private var screenCaptureActive = false
    private var currentBgColor = Color.parseColor("#0F1419")
    private var bgAnimator: ValueAnimator? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        vibrator = getSystemService(Vibrator::class.java)

        viewModel = ViewModelProvider(this)[ConsciousnessViewModel::class.java]

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.state.collect { state ->
                    updateUi(state)
                    processHaptics(state.hapticEvents)
                }
            }
        }

        // Wire touch proprioception bridge BEFORE start() so it's bound when engine initializes
        val touchBridge = SomaTouchBridge()
        touchBridge.updateScreenSize(
            resources.displayMetrics.widthPixels,
            resources.displayMetrics.heightPixels
        )
        viewModel.touchBridge = touchBridge

        viewModel.start(applicationContext)
        // Register bridges AFTER start() creates them — wires lifecycle observers
        viewModel.registerBridges(this)

        // Request runtime permissions for dangerous sensors
        requestSensePermissions()

        // Tap mandala -> UserInput wake signal
        binding.consciousnessMandala.setOnClickListener {
            viewModel.sendWakeSignal(WakeSignal.UserInput)
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        // Long-press mandala -> trigger dream consolidation
        binding.consciousnessMandala.setOnLongClickListener {
            viewModel.dreamConsolidate()
            it.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            true
        }

        // Set up bottom sheet
        setupBottomSheet()

        // Conversation input
        setupConversation()
    }

    private fun setupConversation() {
        val sendAction = {
            val text = binding.conversationInput.text.toString().trim()
            if (text.isNotEmpty()) {
                viewModel.converse(text)
                binding.conversationInput.text.clear()
            }
        }
        binding.btnSend.setOnClickListener { sendAction() }
        binding.conversationInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                sendAction()
                true
            } else false
        }
    }

    private fun setupBottomSheet() {
        val bottomSheet = binding.bottomSheet
        val behavior = BottomSheetBehavior.from(bottomSheet)
        behavior.state = BottomSheetBehavior.STATE_COLLAPSED

        // Dream button
        binding.btnDream.setOnClickListener {
            viewModel.dreamConsolidate()
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        // Vision toggle
        binding.btnVision.setOnClickListener {
            if (screenCaptureActive) {
                viewModel.stopScreenCapture()
                screenCaptureActive = false
                binding.btnVision.text = "enable vision"
                binding.ocrText.visibility = View.GONE
            } else {
                viewModel.screenCapture?.requestPermission(this, SCREEN_CAPTURE_REQUEST)
            }
        }

        // Sleep/Wake toggle
        binding.btnSleep.setOnClickListener {
            if (isAsleep) {
                viewModel.sendWakeSignal(WakeSignal.UserInput)
                isAsleep = false
                binding.btnSleep.text = "sleep"
            } else {
                viewModel.sendWakeSignal(WakeSignal.ExplicitSleep)
                isAsleep = true
                binding.btnSleep.text = "wake"
            }
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        // Checkpoint button
        binding.btnCheckpoint.setOnClickListener {
            viewModel.saveCheckpoint()
            it.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        }

        // Ollama config
        binding.btnOllama.setOnClickListener {
            val host = binding.ollamaHost.text.toString().trim()
            if (host.isNotEmpty()) {
                viewModel.configureOllama(host)
                binding.btnOllama.text = "testing..."
                lifecycleScope.launch {
                    val ok = viewModel.testOllama()
                    binding.btnOllama.text = if (ok) "connected" else "failed"
                }
            }
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

    @Deprecated("Deprecated in API 33, but needed for MediaProjection")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SCREEN_CAPTURE_REQUEST) {
            val granted = viewModel.screenCapture?.onPermissionResult(resultCode, data) ?: false
            if (granted) {
                viewModel.startScreenCapture()
                screenCaptureActive = true
                binding.btnVision.text = "disable vision"
                binding.ocrText.visibility = View.VISIBLE
                binding.ocrText.text = "sees: initializing..."
            }
        }
    }

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        viewModel.touchBridge?.onTouchEvent(event)
        return super.dispatchTouchEvent(event)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            val root = window.decorView
            viewModel.touchBridge?.updateScreenSize(root.width, root.height)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val SCREEN_CAPTURE_REQUEST = 1002
    }

    private fun updateUi(state: SomaUiState) {
        // Update consciousness mandala
        binding.consciousnessMandala.consciousnessLevel = state.consciousnessLevel
        binding.consciousnessMandala.dominantHarmonyColor = harmonyToColor(state.dominantHarmony)

        binding.harmonyText.text = "\u00B7 ${state.dominantHarmony.lowercase()} \u00B7"

        // Animated background gradient shifts with wake state
        val targetBg = when (state.wakeState.lowercase()) {
            "sleep" -> Color.parseColor("#1A1030")     // Deep purple wash
            "drowsy" -> Color.parseColor("#161228")    // Muted purple-dark
            "focused" -> Color.parseColor("#0C1A2A")   // Deep navy
            else -> Color.parseColor("#0F1419")        // Default void
        }
        if (targetBg != currentBgColor) {
            bgAnimator?.cancel()
            bgAnimator = ValueAnimator.ofObject(ArgbEvaluator(), currentBgColor, targetBg).apply {
                duration = 2000L
                addUpdateListener { anim ->
                    val color = anim.animatedValue as Int
                    window.decorView.setBackgroundColor(color)
                    window.statusBarColor = color
                    window.navigationBarColor = color
                }
                start()
            }
            currentBgColor = targetBg
        }

        // Neuromodulator flow view + mandala organic deformation
        val nm = state.neuromodulators
        if (nm.size >= 4) {
            binding.neuromodFlows.levels = floatArrayOf(nm[0], nm[1], nm[2], nm[3])
            binding.consciousnessMandala.neuromodulators = floatArrayOf(nm[0], nm[1], nm[2], nm[3])
        }

        binding.statusText.text =
            "${state.wakeState.lowercase()} \u00B7 ${state.motionState.lowercase()} \u00B7 cycle ${state.cycleCount}"

        // Track sleep state from engine
        isAsleep = state.wakeState.lowercase() == "sleep"
        binding.btnSleep.text = if (isAsleep) "wake" else "sleep"

        binding.dreamText.text =
            "dreams ${state.dreamCount} \u00B7 wisdom ${state.wisdomCount}"

        // Dream narrative — fade in and stay visible (alpha 0.85, not fading out)
        if (state.latestDream.isNotEmpty() && state.latestDream != lastDreamText) {
            lastDreamText = state.latestDream
            binding.dreamNarrative.text = "\u201C${state.latestDream}\u201D"
            binding.dreamNarrative.animate()
                .alpha(0.85f)
                .setDuration(1500)
                .start()
        }

        // Broca voice — show "thinking..." placeholder, then luminous text with longer stay
        if (state.brocaText.isNotEmpty() && state.brocaText != lastBrocaText) {
            lastBrocaText = state.brocaText
            binding.brocaText.text = state.brocaText
            binding.brocaText.setTextColor(Color.parseColor("#00E5CC"))
            binding.brocaText.animate()
                .alpha(1f)
                .setDuration(800)
                .withEndAction {
                    // Fade to readable dim after 8 seconds (not 4s, not 0.4 alpha)
                    binding.brocaText.animate()
                        .alpha(0.6f)
                        .setStartDelay(8000)
                        .setDuration(2000)
                        .start()
                }
                .start()
        }

        binding.peText.text =
            "pe ${"%.2f".format(state.predictionError)} \u00B7 substrate ${"%.2f".format(state.consciousnessLevel)}"

        binding.disclaimerText.text = state.disclaimer.take(80)

        // OCR display when screen capture is active
        if (state.screenCaptureActive && state.ocrText.isNotEmpty()) {
            binding.ocrText.visibility = View.VISIBLE
            binding.ocrText.text = "sees: ${state.ocrText}"
        } else if (!state.screenCaptureActive) {
            binding.ocrText.visibility = View.GONE
        }

        // Update vision button state
        screenCaptureActive = state.screenCaptureActive
        binding.btnVision.text = if (screenCaptureActive) "disable vision" else "enable vision"
    }

    /** Map harmony name to bioluminescent solarpunk palette color. */
    private fun harmonyToColor(harmony: String): Int = when (harmony.lowercase()) {
        "coherence" -> Color.parseColor("#00E5CC")
        "resonance" -> Color.parseColor("#47D4FF")
        "emergence" -> Color.parseColor("#9B7DFF")
        "reciprocity" -> Color.parseColor("#FF7EB3")
        "transparency" -> Color.parseColor("#FFD166")
        "embodiment" -> Color.parseColor("#FF8C42")
        "compassion" -> Color.parseColor("#6BCB77")
        "sacredstillness", "sacred stillness" -> Color.parseColor("#C4B5FD")
        else -> Color.parseColor("#00E5CC")
    }

    /**
     * Dynamic haptic feedback — vibration pattern varies by event type.
     *
     * ConsciousnessShift: gentle pulse (organic bloom)
     * DreamWisdom: double ripple (wisdom surfacing)
     * PeerDiscovered: quick triple-tap (social connection)
     * HighSurprise: sharp flash (attention spike)
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
                // Double ripple: soft on, pause, soft on (wisdom surfacing)
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 50, 120, 50), intArrayOf(0, 100, 0, 80), -1
                ))
            }
            events.contains("PeerDiscovered") -> {
                // Quick triple-tap
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 30, 60, 30, 60, 30), -1
                ))
            }
            events.contains("HighSurprise") -> {
                // Sharp flash — brief high-intensity spike
                v.vibrate(VibrationEffect.createOneShot(50, 255))
            }
            events.contains("HarmonyMilestone") -> {
                // Warm sustained pulse
                v.vibrate(VibrationEffect.createOneShot(150, 128))
            }
            events.contains("ConsciousnessShift") -> {
                // Gentle pulse — organic bloom proportional to shift magnitude
                val amplitude = if (events.contains("0.1") || events.contains("0.2"))
                    180 else 80
                v.vibrate(VibrationEffect.createOneShot(40, amplitude))
            }
            else -> {
                v.vibrate(VibrationEffect.createOneShot(30, 80))
            }
        }
    }
}
