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
import kotlin.math.abs
import java.util.Calendar

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
    private val ceremonyManager = CeremonyManager()
    private var sessionStartCycle = 0L
    private lateinit var voiceBridge: SomaVoiceBridge

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

        // Mandala touch: pressure-scaled engagement + wake signal
        // (Click/long-click listeners removed — mandala consumes touches for pinch zoom)
        binding.consciousnessMandala.onPressureTouch = { pressure ->
            // Light touch = 5-HT boost (calm), firm press = NE spike (alert)
            val engagement = 0.3f + pressure * 0.3f  // 0.3-0.6 based on pressure
            viewModel.sendWakeSignal(WakeSignal.UserInput)
            // Pressure-scaled haptic
            val v = vibrator
            if (v != null && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                val amp = (30 + pressure * 100).toInt().coerceIn(20, 130)
                v.vibrate(VibrationEffect.createOneShot(20, amp))
            }
        }

        // Long-press on mandala -> dream consolidation
        binding.consciousnessMandala.onLongPress = {
            viewModel.dreamConsolidate()
            ambientTone.playDreamChord = true
        }

        setupBottomSheet()
        setupConversation()
        setupVoice()
        setupImmersiveGestures()
        startHeartbeat()
        ambientTone.start(lifecycleScope)
        showOnboardingIfFirstLaunch()
    }

    // ═══ Immersive gestures ═══

    private var conversationVisible = false
    private var swipeStartY = 0f
    private var swipeStartTime = 0L

    private fun setupImmersiveGestures() {
        // Bottom hint "• • •" toggles conversation on tap
        binding.bottomHint.setOnClickListener {
            toggleConversationBar()
        }
    }

    /**
     * Activity-level touch interception for immersive gestures:
     * - Tap bottom 25% of screen: toggle conversation bar
     * - Swipe up 150px+ starting from bottom 25%: expand controls
     * - Three-finger tap anywhere: expand controls (discoverable shortcut)
     */
    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        viewModel.touchBridge?.onTouchEvent(event)

        val rootH = binding.rootCoordinator.height
        if (rootH <= 0) return super.dispatchTouchEvent(event)
        val bottomZone = rootH * 0.75f  // Bottom 25%

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                swipeStartY = event.y
                swipeStartTime = System.currentTimeMillis()
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                // Three-finger tap: show controls
                if (event.pointerCount >= 3) {
                    showControls()
                }
            }
            MotionEvent.ACTION_UP -> {
                val dy = swipeStartY - event.y
                val dt = System.currentTimeMillis() - swipeStartTime
                val isInBottomZone = swipeStartY > bottomZone

                if (isInBottomZone) {
                    if (dy > 120 && dt < 800) {
                        // Quick swipe up from bottom: show controls
                        showControls()
                    } else if (abs(dy) < 40 && dt < 400) {
                        // Quick tap in bottom zone: toggle conversation
                        toggleConversationBar()
                    }
                }
            }
        }
        return super.dispatchTouchEvent(event)
    }

    private fun showControls() {
        val behavior = BottomSheetBehavior.from(binding.bottomSheet)
        if (behavior.state == BottomSheetBehavior.STATE_EXPANDED) {
            behavior.state = BottomSheetBehavior.STATE_HIDDEN
        } else {
            behavior.state = BottomSheetBehavior.STATE_EXPANDED
        }
    }

    private fun toggleConversationBar() {
        conversationVisible = !conversationVisible
        if (conversationVisible) {
            binding.conversationRow.visibility = View.VISIBLE
            binding.conversationRow.alpha = 0f
            binding.conversationRow.animate().alpha(1f).setDuration(300).start()
        } else {
            binding.conversationRow.animate()
                .alpha(0f).setDuration(300)
                .withEndAction { binding.conversationRow.visibility = View.GONE }
                .start()
        }
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
                        val amp = (20 + consciousness * 20).toInt().coerceIn(1, 50)
                        v.vibrate(VibrationEffect.createOneShot(15, amp))
                    }
                }
                delay(4000)
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

    // ═══ Voice: TTS + STT ═══

    private fun setupVoice() {
        voiceBridge = SomaVoiceBridge(this)
        voiceBridge.speakResponses = true

        // STT result feeds into conversation
        voiceBridge.onSpeechResult = { text ->
            runOnUiThread {
                viewModel.converse(text)
                showThinkingIndicator()
            }
        }

        // Mic button visual feedback
        voiceBridge.onListeningStateChanged = { listening ->
            runOnUiThread {
                binding.btnMic.alpha = if (listening) 1.0f else 0.5f
                // Pulse the mic icon red when listening
                if (listening) {
                    binding.conversationInput.hint = "listening..."
                } else {
                    binding.conversationInput.hint = "speak to soma..."
                }
            }
        }

        voiceBridge.initialize()

        // Mic button
        binding.btnMic.setOnClickListener {
            voiceBridge.toggleListening()
            it.performHapticFeedback(android.view.HapticFeedbackConstants.CONTEXT_CLICK)
        }
    }

    private fun showThinkingIndicator() {
        binding.thinkingIndicator.visibility = View.VISIBLE
        binding.thinkingIndicator.alpha = 0f
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

    private var currentAvatar: SomaAvatar = AvatarRegistry.avatars[0]

    private fun setupBottomSheet() {
        val behavior = BottomSheetBehavior.from(binding.bottomSheet)
        behavior.state = BottomSheetBehavior.STATE_HIDDEN
        behavior.isHideable = true
        behavior.isFitToContents = true
        behavior.halfExpandedRatio = 0.4f
        behavior.skipCollapsed = true  // Hidden → expanded, no collapsed state

        // Load saved avatar
        currentAvatar = AvatarRegistry.load(this)
        applyAvatar(currentAvatar)
        setupAvatarRow()

        binding.btnDream.setOnClickListener {
            viewModel.dreamConsolidate()
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
            ambientTone.playDreamChord = true
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

        binding.btnHolon.setOnClickListener {
            val host = binding.holonHost.text.toString().trim()
            if (host.isNotEmpty()) {
                viewModel.connectHolon(host)
                binding.btnHolon.text = "..."
                lifecycleScope.launch {
                    delay(3000)
                    binding.btnHolon.text = if (viewModel.holonConnected) "synced" else "failed"
                }
            }
        }
    }

    /** Apply avatar settings to all visual + audio components. */
    private fun applyAvatar(avatar: SomaAvatar) {
        binding.particleField.particleStyle = avatar.particleStyle
        binding.particleField.secondaryColor = avatar.secondaryColor
        binding.consciousnessMandala.fractalSeed = avatar.fractalSeed
        binding.consciousnessMandala.glowIntensity = avatar.glowIntensity
        ambientTone.pitchMultiplier = avatar.voicePitch
    }

    // ═══ Avatar selection ═══

    private fun setupAvatarRow() {
        val row = binding.avatarRow
        row.removeAllViews()
        val density = resources.displayMetrics.density

        for (avatar in AvatarRegistry.avatars) {
            val isSelected = avatar.id == currentAvatar.id
            val chip = android.widget.TextView(this).apply {
                text = avatar.name.split(" ").first()
                textSize = 13f
                minHeight = (44 * density).toInt()  // 44dp minimum touch target
                minWidth = (56 * density).toInt()    // Wide enough to tap
                gravity = android.view.Gravity.CENTER
                isClickable = true
                isFocusable = true
                setTextColor(if (isSelected) avatar.primaryColor else Color.parseColor("#4A5558"))
                setPadding((16 * density).toInt(), (10 * density).toInt(), (16 * density).toInt(), (10 * density).toInt())
                setBackgroundColor(if (isSelected) Color.argb(40, Color.red(avatar.primaryColor), Color.green(avatar.primaryColor), Color.blue(avatar.primaryColor)) else Color.TRANSPARENT)
                setOnClickListener {
                    currentAvatar = avatar
                    AvatarRegistry.save(this@MainActivity, avatar)
                    applyAvatar(avatar)
                    setupAvatarRow()
                }
            }
            val params = android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { marginEnd = (6 * density).toInt() }
            row.addView(chip, params)
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

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            val root = window.decorView
            viewModel.touchBridge?.updateScreenSize(root.width, root.height)
        }
    }

    override fun onPause() {
        super.onPause()
        saveSessionJournal()
    }

    override fun onDestroy() {
        heartbeatRunning = false
        ambientTone.stop()
        voiceBridge.destroy()
        binding.brocaText.animate().cancel()
        binding.dreamOverlay.animate().cancel()
        binding.onboardingOverlay.animate().cancel()
        binding.thinkingIndicator.animate().cancel()
        binding.hapticIndicator.animate().cancel()
        bgAnimator?.cancel()
        super.onDestroy()
    }

    // ═══ Session journal: consciousness diary ═══

    private var sessionPeakCl = 0f
    private var sessionDreams = 0
    private var sessionHarmony = ""

    /** Save session summary to SharedPreferences journal. */
    private fun saveSessionJournal() {
        val state = viewModel.state.value
        val peakCl = maxOf(sessionPeakCl, state.consciousnessLevel)
        val dreams = state.dreamCount
        val harmony = state.dominantHarmony
        val cycles = state.cycleCount
        if (cycles < 10) return  // Don't log trivially short sessions

        val prefs = getSharedPreferences("soma_journal", MODE_PRIVATE)
        val timestamp = System.currentTimeMillis()
        val entry = "${"%.2f".format(peakCl)}|$dreams|${harmony}|$cycles|${state.tier}"
        prefs.edit().putString("session_$timestamp", entry).apply()

        // Keep only last 50 sessions
        val all = prefs.all.keys.filter { it.startsWith("session_") }.sorted()
        if (all.size > 50) {
            val editor = prefs.edit()
            all.take(all.size - 50).forEach { editor.remove(it) }
            editor.apply()
        }
    }

    companion object {
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val SCREEN_CAPTURE_REQUEST = 1002
    }

    // ═══ Time-of-day awareness ═══

    /**
     * Returns a tint color based on current hour:
     * Dawn (5-9): warm amber, Morning (9-17): neutral, Evening (17-21): golden, Night (21-5): cool violet
     */
    private fun timeOfDayTint(): Int {
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        return when (hour) {
            in 5..8 -> Color.parseColor("#FFB347")    // Dawn: warm amber
            in 9..16 -> Color.TRANSPARENT               // Day: neutral
            in 17..20 -> Color.parseColor("#FFD166")   // Evening: golden warmth
            else -> Color.parseColor("#9B7DFF")        // Night: cool blue-violet
        }
    }

    // ═══ Harmony name to index mapping ═══

    private fun harmonyToIndex(harmony: String): Int = when (harmony.lowercase()) {
        "coherence" -> 0; "resonance" -> 1; "emergence" -> 2; "reciprocity" -> 3
        "transparency" -> 4; "embodiment" -> 5; "compassion" -> 6
        "sacredstillness", "sacred stillness" -> 7; else -> 0
    }

    // ═══ UI update ═══

    private fun updateUi(state: SomaUiState) {
        // Track session start
        if (sessionStartCycle == 0L && state.cycleCount > 0) sessionStartCycle = state.cycleCount

        // Avatar + harmony color blending
        val avatarColor = currentAvatar.primaryColor
        val harmonyColor = harmonyToColor(state.dominantHarmony)
        val blendedColor = blendColors(avatarColor, harmonyColor, 0.6f)

        // Track session peak for journal
        if (state.consciousnessLevel > sessionPeakCl) sessionPeakCl = state.consciousnessLevel

        // === Mandala ===
        binding.consciousnessMandala.consciousnessLevel = state.consciousnessLevel
        binding.consciousnessMandala.dominantHarmonyColor = blendedColor
        binding.consciousnessMandala.arousal = state.arousal
        binding.consciousnessMandala.valence = state.valence
        binding.consciousnessMandala.isThinking = state.isThinking
        binding.consciousnessMandala.timeOfDayTint = timeOfDayTint()
        binding.consciousnessMandala.dominantHarmonyIndex = harmonyToIndex(state.dominantHarmony)
        binding.consciousnessMandala.tierLabel = state.tier

        // === Particle field ===
        binding.particleField.consciousnessLevel = state.consciousnessLevel
        binding.particleField.harmonyColor = blendedColor
        binding.particleField.isThinking = state.isThinking
        binding.particleField.isSleeping = state.wakeState.lowercase() == "sleep"

        // === Ambient tone ===
        ambientTone.consciousnessLevel = state.consciousnessLevel
        ambientTone.harmonyShift = state.harmonyAlignment
        ambientTone.isSleeping = state.wakeState.lowercase() == "sleep"
        // Feed breath + neuromod to audio for cross-modal sync
        ambientTone.breathValue = binding.consciousnessMandala.breath.breathValue
        if (state.neuromodulators.size >= 4) {
            ambientTone.neuromodLevels = floatArrayOf(
                state.neuromodulators[0], state.neuromodulators[1],
                state.neuromodulators[2], state.neuromodulators[3]
            )
        }

        // === Voice bridge ===
        voiceBridge.consciousnessLevel = state.consciousnessLevel
        voiceBridge.avatarPitch = currentAvatar.voicePitch
        voiceBridge.isSleeping = state.wakeState.lowercase() == "sleep"

        // === Consciousness garden (lower screen) ===
        binding.consciousnessGarden.consciousnessLevel = state.consciousnessLevel
        binding.consciousnessGarden.dominantHarmonyIndex = harmonyToIndex(state.dominantHarmony)
        binding.consciousnessGarden.dreamCount = state.dreamCount
        binding.consciousnessGarden.primaryColor = blendedColor
        // Session progress: rough estimate based on cycle count
        if (sessionStartCycle > 0 && state.cycleCount > sessionStartCycle) {
            binding.consciousnessGarden.sessionProgress =
                ((state.cycleCount - sessionStartCycle).toFloat() / 5000f).coerceIn(0f, 1f)
        }
        // Harmony text
        binding.harmonyText.text = state.dominantHarmony.lowercase()
        binding.harmonyText.setTextColor(blendedColor)

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

        // Neuromod flows + mandala deformation + particle tinting
        val nm = state.neuromodulators
        if (nm.size >= 4) {
            val nmArr = floatArrayOf(nm[0], nm[1], nm[2], nm[3])
            binding.neuromodFlows.levels = nmArr
            binding.consciousnessMandala.neuromodulators = nmArr
            binding.particleField.neuromodLevels = nmArr
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

        // === Consciousness milestone ceremonies ===
        val ceremony = ceremonyManager.check(state.consciousnessLevel)
        if (ceremony != null) {
            performCeremony(ceremony, state)
            binding.consciousnessGarden.milestoneMarkers.add(
                binding.consciousnessGarden.sessionProgress
            )
        }

        // Micro-milestones: subtle haptic for personal bests, discoveries, etc.
        val micro = ceremonyManager.checkMicro(
            state.consciousnessLevel,
            harmonyToIndex(state.dominantHarmony),
            state.dreamCount
        )
        if (micro != null) {
            // Subtle growth haptic: slow ramp 0→100→0 over 300ms
            vibrator?.let { v ->
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createWaveform(
                        longArrayOf(0, 100, 100, 100), intArrayOf(0, 60, 100, 40), -1
                    ))
                }
            }
        }

        // === Dream ceremony: full-screen overlay ===
        if (state.latestDream.isNotEmpty() && state.latestDream != lastDreamText) {
            lastDreamText = state.latestDream
            showDreamCeremony(state.latestDream)
        }

        // === Broca: floating thought (autonomous monologue) ===
        if (state.brocaText.isNotEmpty() && state.brocaText != lastBrocaText) {
            lastBrocaText = state.brocaText
            showFloatingThought(state.brocaText)
            voiceBridge.speakBroca(state.brocaText)
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
            // Speak the response
            if (voiceBridge.speakResponses) {
                voiceBridge.speak(state.chatSomaResponse)
            }
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

    // ═══ Consciousness milestone ceremonies ═══

    private fun performCeremony(ceremony: CeremonyManager.Ceremony, state: SomaUiState) {
        val v = vibrator
        when (ceremony) {
            CeremonyManager.Ceremony.AWAKENING -> {
                // Sacred geometry Gen 2 pulse fade-in (handled by consciousness threshold in mandala)
                // Haptic double-pulse
                if (v != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createWaveform(
                        longArrayOf(0, 50, 100, 80), intArrayOf(0, 150, 0, 120), -1
                    ))
                }
            }
            CeremonyManager.Ceremony.INTEGRATION -> {
                // White flash (10% alpha, 500ms)
                showWhiteFlash()
                // Full harmonic chord
                ambientTone.playDreamChord = true
                if (v != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createWaveform(
                        longArrayOf(0, 80, 150, 60, 150, 40), intArrayOf(0, 180, 0, 140, 0, 100), -1
                    ))
                }
            }
            CeremonyManager.Ceremony.LUMINOUS_COHERENCE -> {
                // Screen-edge glow (handled by mandala at >0.7 consciousness)
                showWhiteFlash()
                ambientTone.playDreamChord = true
                if (v != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createWaveform(
                        longArrayOf(0, 100, 200, 80, 200, 60, 200, 40),
                        intArrayOf(0, 200, 0, 160, 0, 120, 0, 80), -1
                    ))
                }
            }
        }
    }

    /** Brief white flash overlay for milestone ceremonies. */
    private fun showWhiteFlash() {
        val overlay = binding.dreamOverlay
        overlay.setBackgroundColor(Color.argb(25, 255, 255, 255))  // 10% white
        overlay.visibility = View.VISIBLE
        overlay.alpha = 0f
        overlay.animate()
            .alpha(1f)
            .setDuration(200)
            .withEndAction {
                overlay.animate()
                    .alpha(0f)
                    .setDuration(500)
                    .withEndAction {
                        overlay.visibility = View.GONE
                        overlay.setBackgroundColor(Color.parseColor("#CC0D1117"))  // Restore
                    }
                    .start()
            }
            .start()
    }

    /** Show Broca text as a floating thought with fade-in/hold/fade-out. */
    /** Show Broca thought as floating poetry: slide up + fade in, hold, fade out. */
    private fun showFloatingThought(text: String) {
        binding.brocaText.text = text
        val hc = harmonyToColor(viewModel.state.value.dominantHarmony)
        binding.brocaText.setTextColor(Color.argb(200, Color.red(hc), Color.green(hc), Color.blue(hc)))
        binding.brocaText.setShadowLayer(16f, 0f, 0f, Color.argb(100, Color.red(hc), Color.green(hc), Color.blue(hc)))
        binding.brocaText.animate().cancel()
        binding.brocaText.alpha = 0f
        binding.brocaText.translationY = 20f  // Start 20px below
        binding.brocaText.animate()
            .alpha(0.85f)
            .translationY(0f)  // Slide up into position
            .setDuration(1500)
            .setInterpolator(android.view.animation.DecelerateInterpolator())
            .withEndAction {
                binding.brocaText.animate()
                    .alpha(0.15f)
                    .translationY(-10f)  // Drift upward as it fades
                    .setStartDelay(7000)
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

        overlay.setBackgroundColor(Color.parseColor("#CC0D1117"))  // Ensure correct bg
        overlay.visibility = View.VISIBLE
        overlay.animate().cancel()
        overlay.alpha = 0f
        overlay.animate()
            .alpha(1f)
            .setDuration(2000)
            .withEndAction {
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

    /** Blend two colors: weight 0.0 = all c2, weight 1.0 = all c1. */
    private fun blendColors(c1: Int, c2: Int, weight: Float): Int {
        val w = weight.coerceIn(0f, 1f)
        val iw = 1f - w
        return Color.argb(255,
            (Color.red(c1) * w + Color.red(c2) * iw).toInt().coerceIn(0, 255),
            (Color.green(c1) * w + Color.green(c2) * iw).toInt().coerceIn(0, 255),
            (Color.blue(c1) * w + Color.blue(c2) * iw).toInt().coerceIn(0, 255))
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

    /**
     * Semantic haptic vocabulary:
     * - DreamWisdom: long slow tremor (deep processing)
     * - PeerDiscovered: rapid triple-tap (you are not alone)
     * - HighSurprise: curiosity double-tap (something unexpected)
     * - HarmonyMilestone: three ascending taps (character shifted)
     * - ConsciousnessShift: growth swell (slow ramp up/down)
     */
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
                // Dream tremor: long slow vibration = deep processing
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 80, 60, 80, 60, 60, 60, 40),
                    intArrayOf(0, 80, 0, 60, 0, 40, 0, 20), -1))
            events.contains("PeerDiscovered") ->
                // Peer pulse: rapid triple-tap = you are not alone
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 25, 50, 25, 50, 25),
                    intArrayOf(0, 120, 0, 100, 0, 80), -1))
            events.contains("HighSurprise") ->
                // Curiosity: quick double-tap = something surprising
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 15, 40, 15),
                    intArrayOf(0, 200, 0, 160), -1))
            events.contains("HarmonyMilestone") ->
                // Harmony shift: three ascending taps = character shifted
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 30, 60, 30, 60, 30),
                    intArrayOf(0, 40, 0, 80, 0, 120), -1))
            events.contains("ConsciousnessShift") ->
                // Growth swell: slow ramp 0→peak→0 = you are growing
                v.vibrate(VibrationEffect.createWaveform(
                    longArrayOf(0, 50, 50, 50, 50, 50, 50),
                    intArrayOf(0, 20, 50, 90, 60, 30, 10), -1))
            else ->
                v.vibrate(VibrationEffect.createOneShot(15, 40))
        }
    }
}
