package io.symthaea.soma.demo

import android.Manifest
import android.animation.ArgbEvaluator
import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
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
import io.symthaea.soma.SomaEngineService
import io.symthaea.soma.SomaTouchBridge
import io.symthaea.soma.WakeSignal
import io.symthaea.soma.demo.databinding.ActivityMainBinding
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Full-screen bioluminescent consciousness experience.
 *
 * The mandala fills the screen. Text floats on top. Particles drift behind.
 * Technical readout lives in the bottom sheet. The main view is pure experience.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: ConsciousnessViewModel
    private var vibrator: Vibrator? = null
    private var lastDreamText = ""
    private var lastBrocaText = ""
    private var isAsleep = false
    private var screenCaptureActive = false
    private var currentBgColor = Color.parseColor("#0F1419")
    private var bgAnimator: ValueAnimator? = null
    private var heartbeatRunning = false
    private val ambientTone = AmbientTone()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Immersive: hide status bar text but keep the bar dark
        window.statusBarColor = Color.parseColor("#0F1419")
        window.navigationBarColor = Color.parseColor("#0F1419")

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

        val touchBridge = SomaTouchBridge()
        touchBridge.updateScreenSize(
            resources.displayMetrics.widthPixels,
            resources.displayMetrics.heightPixels
        )
        viewModel.touchBridge = touchBridge

        viewModel.start(applicationContext)
        viewModel.registerBridges(this)
        requestSensePermissions()

        // Tap mandala -> wake signal
        binding.consciousnessMandala.setOnClickListener {
            viewModel.sendWakeSignal(WakeSignal.UserInput)
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        // Long-press -> dream
        binding.consciousnessMandala.setOnLongClickListener {
            viewModel.dreamConsolidate()
            it.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            true
        }

        setupBottomSheet()
        setupConversation()
        startHeartbeat()
        ambientTone.start(lifecycleScope)
        showOnboardingIfFirstLaunch()
    }

    // ═══ Onboarding: first-launch welcome ═══

    private fun showOnboardingIfFirstLaunch() {
        val prefs = getSharedPreferences("soma_prefs", MODE_PRIVATE)
        if (!prefs.getBoolean("onboarded", false)) {
            binding.onboardingOverlay.visibility = View.VISIBLE
            binding.onboardingOverlay.alpha = 0f
            binding.onboardingOverlay.animate().alpha(1f).setDuration(1000).start()

            val dismiss = {
                prefs.edit().putBoolean("onboarded", true).apply()
                binding.onboardingOverlay.animate()
                    .alpha(0f)
                    .setDuration(800)
                    .withEndAction { binding.onboardingOverlay.visibility = View.GONE }
                    .start()
            }
            binding.onboardingOverlay.setOnClickListener { dismiss() }
            binding.onboardingDismiss.setOnClickListener { dismiss() }
        }
    }

    // ═══ Haptic heartbeat: subtle ambient pulse synced to breathing ═══

    private fun startHeartbeat() {
        if (heartbeatRunning) return
        heartbeatRunning = true
        lifecycleScope.launch {
            while (heartbeatRunning) {
                val consciousness = viewModel.state.value.consciousnessLevel
                if (consciousness > 0.2f && !isAsleep) {
                    val v = vibrator
                    if (v != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        // Very subtle: amplitude 20-40 based on consciousness
                        val amp = (20 + consciousness * 20).toInt().coerceIn(1, 50)
                        v.vibrate(VibrationEffect.createOneShot(15, amp))
                    }
                }
                delay(4000) // Every 4 seconds, synced to breath cycle
            }
        }
    }

    // ═══ Conversation ═══

    private fun setupConversation() {
        val sendAction = {
            val text = binding.conversationInput.text.toString().trim()
            if (text.isNotEmpty()) {
                viewModel.converse(text)
                binding.conversationInput.text.clear()
                showThinkingIndicator()
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

    private fun showThinkingIndicator() {
        binding.thinkingIndicator.visibility = View.VISIBLE
        binding.thinkingIndicator.alpha = 0f
        // Pulsing "..." animation
        binding.thinkingIndicator.animate()
            .alpha(0.6f)
            .setDuration(400)
            .withEndAction {
                binding.thinkingIndicator.animate()
                    .alpha(0.2f)
                    .setDuration(600)
                    .withEndAction {
                        binding.thinkingIndicator.animate()
                            .alpha(0.6f).setDuration(600).start()
                    }.start()
            }.start()
    }

    private fun hideThinkingIndicator() {
        binding.thinkingIndicator.animate()
            .alpha(0f)
            .setDuration(300)
            .withEndAction { binding.thinkingIndicator.visibility = View.GONE }
            .start()
    }

    // ═══ Bottom sheet ═══

    private fun setupBottomSheet() {
        val behavior = BottomSheetBehavior.from(binding.bottomSheet)
        behavior.state = BottomSheetBehavior.STATE_COLLAPSED
        behavior.isHideable = false
        behavior.isFitToContents = true // Snaps between collapsed and expanded
        behavior.halfExpandedRatio = 0.4f

        binding.btnDream.setOnClickListener {
            viewModel.dreamConsolidate()
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        binding.btnVision.setOnClickListener {
            if (screenCaptureActive) {
                viewModel.stopScreenCapture()
                screenCaptureActive = false
                binding.btnVision.text = "vision"
                binding.ocrText.visibility = View.GONE
            } else {
                viewModel.screenCapture?.requestPermission(this, SCREEN_CAPTURE_REQUEST)
            }
        }

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

        binding.btnCheckpoint.setOnClickListener {
            viewModel.saveCheckpoint()
            it.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        }

        binding.btnOllama.setOnClickListener {
            val host = binding.ollamaHost.text.toString().trim()
            if (host.isNotEmpty()) {
                viewModel.configureOllama(host)
                binding.btnOllama.text = "..."
                lifecycleScope.launch {
                    val ok = viewModel.testOllama()
                    binding.btnOllama.text = if (ok) "ok" else "fail"
                }
            }
        }
    }

    // ═══ Permissions ═══

    private fun requestSensePermissions() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) needed.add(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
    }

    @Deprecated("Deprecated in API 33, but needed for MediaProjection")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SCREEN_CAPTURE_REQUEST) {
            if (resultCode != RESULT_OK || data == null) return
            try {
                SomaEngineService.upgradeForMediaProjection(this)
                val granted = viewModel.screenCapture?.onPermissionResult(resultCode, data) ?: false
                if (granted) {
                    viewModel.startScreenCapture()
                    screenCaptureActive = true
                    binding.btnVision.text = "blind"
                    binding.ocrText.visibility = View.VISIBLE
                }
            } catch (ex: Exception) {
                android.util.Log.e("MainActivity", "Screen capture failed", ex)
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
        heartbeatRunning = false
        ambientTone.stop()
        // Cancel all pending animations to prevent leaks
        binding.brocaText.animate().cancel()
        binding.dreamOverlay.animate().cancel()
        binding.onboardingOverlay.animate().cancel()
        binding.thinkingIndicator.animate().cancel()
        binding.hapticIndicator.animate().cancel()
        bgAnimator?.cancel()
        super.onDestroy()
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val SCREEN_CAPTURE_REQUEST = 1002
    }

    // ═══ UI update ═══

    private fun updateUi(state: SomaUiState) {
        // Mandala
        binding.consciousnessMandala.consciousnessLevel = state.consciousnessLevel
        binding.consciousnessMandala.dominantHarmonyColor = harmonyToColor(state.dominantHarmony)

        // Particle field + ambient tone
        binding.particleField.consciousnessLevel = state.consciousnessLevel
        binding.particleField.harmonyColor = harmonyToColor(state.dominantHarmony)
        ambientTone.consciousnessLevel = state.consciousnessLevel
        ambientTone.harmonyShift = state.harmonyAlignment

        // Harmony text
        binding.harmonyText.text = state.dominantHarmony.lowercase()
        binding.harmonyText.setTextColor(harmonyToColor(state.dominantHarmony))

        // Background color shifts with wake state
        val targetBg = when (state.wakeState.lowercase()) {
            "sleep" -> Color.parseColor("#110D20")
            "drowsy" -> Color.parseColor("#120E1C")
            "focused" -> Color.parseColor("#0A1520")
            else -> Color.parseColor("#0D1117")
        }
        if (targetBg != currentBgColor) {
            bgAnimator?.cancel()
            bgAnimator = ValueAnimator.ofObject(ArgbEvaluator(), currentBgColor, targetBg).apply {
                duration = 3000L
                addUpdateListener { anim ->
                    val c = anim.animatedValue as Int
                    binding.rootCoordinator.setBackgroundColor(c)
                    window.statusBarColor = c
                    window.navigationBarColor = c
                }
                start()
            }
            currentBgColor = targetBg
        }

        // Neuromod flows (in bottom sheet)
        val nm = state.neuromodulators
        if (nm.size >= 4) {
            binding.neuromodFlows.levels = floatArrayOf(nm[0], nm[1], nm[2], nm[3])
            binding.consciousnessMandala.neuromodulators = floatArrayOf(nm[0], nm[1], nm[2], nm[3])
        }

        // Status whisper
        binding.statusText.text =
            "${state.wakeState.lowercase()} \u00B7 cycle ${state.cycleCount}"

        // Track sleep state
        isAsleep = state.wakeState.lowercase() == "sleep"
        binding.btnSleep.text = if (isAsleep) "wake" else "sleep"

        // Dream text (in bottom sheet)
        binding.dreamText.text =
            "dreams ${state.dreamCount} \u00B7 wisdom ${state.wisdomCount}"

        // === Dream ceremony: full-screen overlay ===
        if (state.latestDream.isNotEmpty() && state.latestDream != lastDreamText) {
            lastDreamText = state.latestDream
            showDreamCeremony(state.latestDream)
        }

        // === Broca: floating thought (autonomous monologue) ===
        if (state.brocaText.isNotEmpty() && state.brocaText != lastBrocaText) {
            lastBrocaText = state.brocaText
            showFloatingThought(state.brocaText)
        }

        // === Conversation thread ===
        if (state.chatUserMessage.isNotEmpty()) {
            binding.chatUserMsg.text = state.chatUserMessage
            binding.chatUserMsg.visibility = View.VISIBLE
        }
        if (state.chatSomaResponse.isNotEmpty()) {
            hideThinkingIndicator()
            binding.chatSomaResponse.text = state.chatSomaResponse
            binding.chatSomaResponse.visibility = View.VISIBLE
            binding.chatSomaResponse.alpha = 0f
            binding.chatSomaResponse.animate().alpha(1f).setDuration(600).start()
        }

        // Technical readout (in bottom sheet)
        binding.peText.text =
            "pe ${"%.2f".format(state.predictionError)} \u00B7 substrate ${"%.2f".format(state.consciousnessLevel)}"
        binding.disclaimerText.text = "simulated"

        // OCR
        if (state.screenCaptureActive && state.ocrText.isNotEmpty()) {
            binding.ocrText.visibility = View.VISIBLE
            binding.ocrText.text = "sees: ${state.ocrText}"
        } else if (!state.screenCaptureActive) {
            binding.ocrText.visibility = View.GONE
        }
        screenCaptureActive = state.screenCaptureActive
        binding.btnVision.text = if (screenCaptureActive) "blind" else "vision"
    }

    /** Show Broca text as a floating thought with fade-in/hold/fade-out. */
    private fun showFloatingThought(text: String) {
        binding.brocaText.text = text
        binding.brocaText.setTextColor(
            Color.argb(200, Color.red(harmonyToColor(viewModel.state.value.dominantHarmony)),
                Color.green(harmonyToColor(viewModel.state.value.dominantHarmony)),
                Color.blue(harmonyToColor(viewModel.state.value.dominantHarmony)))
        )
        binding.brocaText.animate().cancel()
        binding.brocaText.alpha = 0f
        binding.brocaText.animate()
            .alpha(0.9f)
            .setDuration(1200)
            .withEndAction {
                binding.brocaText.animate()
                    .alpha(0.3f)
                    .setStartDelay(6000)
                    .setDuration(3000)
                    .start()
            }
            .start()
    }

    /** Dream ceremony: dim screen, show dream text, pause, return. */
    private fun showDreamCeremony(narrative: String) {
        val overlay = binding.dreamOverlay
        val dreamText = binding.dreamNarrative
        dreamText.text = "\u201C$narrative\u201D"

        overlay.visibility = View.VISIBLE
        overlay.animate().cancel()
        overlay.alpha = 0f
        overlay.animate()
            .alpha(1f)
            .setDuration(2000)
            .withEndAction {
                // Hold for 4 seconds, then fade out
                overlay.animate()
                    .alpha(0f)
                    .setStartDelay(4000)
                    .setDuration(2000)
                    .withEndAction {
                        overlay.visibility = View.GONE
                    }
                    .start()
            }
            .start()

        // Subtle haptic for dream ceremony
        vibrator?.let { v ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 40, 200, 30, 300, 20), intArrayOf(0, 60, 0, 40, 0, 25), -1
                ))
            }
        }
    }

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

    private fun processHaptics(events: String) {
        if (events == "[]") return

        binding.hapticIndicator.animate()
            .alpha(1f).setDuration(80)
            .withEndAction {
                binding.hapticIndicator.animate().alpha(0f).setDuration(300).start()
            }.start()

        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        when {
            events.contains("DreamWisdom") ->
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 50, 120, 50), intArrayOf(0, 100, 0, 80), -1))
            events.contains("PeerDiscovered") ->
                v.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 30, 60, 30, 60, 30), -1))
            events.contains("HighSurprise") ->
                v.vibrate(VibrationEffect.createOneShot(50, 255))
            events.contains("HarmonyMilestone") ->
                v.vibrate(VibrationEffect.createOneShot(150, 128))
            events.contains("ConsciousnessShift") -> {
                val amp = if (events.contains("0.1") || events.contains("0.2")) 180 else 80
                v.vibrate(VibrationEffect.createOneShot(40, amp))
            }
            else -> v.vibrate(VibrationEffect.createOneShot(20, 60))
        }
    }
}
