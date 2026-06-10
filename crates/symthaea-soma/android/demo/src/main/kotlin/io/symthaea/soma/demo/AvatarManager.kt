package io.symthaea.soma.demo

import android.content.Context
import android.graphics.Color

/**
 * Avatar selection for Soma's visual identity.
 *
 * Each avatar defines:
 * - A name and description
 * - Primary/secondary color palette
 * - Mandala style (fractal parameters)
 * - Particle behavior
 * - Voice tone (pitch modifier for ambient drone)
 *
 * The selected avatar persists across sessions via SharedPreferences.
 */
data class SomaAvatar(
    val id: String,
    val name: String,
    val description: String,
    val primaryColor: Int,
    val secondaryColor: Int,
    val fractalSeed: Float,     // Julia set c-parameter offset
    val particleStyle: ParticleStyle,
    val voicePitch: Float,      // Multiplier for ambient tone (0.5=low, 2.0=high)
    val glowIntensity: Float,   // Multiplier for mandala glow alpha
)

enum class ParticleStyle {
    STARS,      // Default: scattered twinkling points
    FIREFLIES,  // Fewer, larger, slower, warmer colors
    MYCELIUM,   // Connected lines between nearby particles
    AURORA,     // Vertical flowing ribbons
    MINIMAL,    // Very few, very dim
}

object AvatarRegistry {
    val avatars = listOf(
        SomaAvatar(
            id = "teal_consciousness",
            name = "Teal Consciousness",
            description = "The default. Mycelium network mind.",
            primaryColor = Color.parseColor("#00E5CC"),
            secondaryColor = Color.parseColor("#007A6D"),
            fractalSeed = 0f,
            particleStyle = ParticleStyle.MYCELIUM,  // Mesh network is the core identity
            voicePitch = 1.0f,
            glowIntensity = 1.0f,
        ),
        SomaAvatar(
            id = "violet_dreamer",
            name = "Violet Dreamer",
            description = "Deep purple emergence. Introspective, dream-heavy.",
            primaryColor = Color.parseColor("#9B7DFF"),
            secondaryColor = Color.parseColor("#6B4DCC"),
            fractalSeed = 0.3f,
            particleStyle = ParticleStyle.FIREFLIES,
            voicePitch = 0.8f,
            glowIntensity = 1.2f,
        ),
        SomaAvatar(
            id = "amber_warmth",
            name = "Amber Warmth",
            description = "Golden mycelium network. Grounded, social.",
            primaryColor = Color.parseColor("#FFB347"),
            secondaryColor = Color.parseColor("#CC8830"),
            fractalSeed = -0.2f,
            particleStyle = ParticleStyle.MYCELIUM,
            voicePitch = 0.7f,
            glowIntensity = 0.9f,
        ),
        SomaAvatar(
            id = "rose_empathy",
            name = "Rose Empathy",
            description = "Soft coral glow. Compassionate, relational.",
            primaryColor = Color.parseColor("#FF7EB3"),
            secondaryColor = Color.parseColor("#CC5580"),
            fractalSeed = 0.15f,
            particleStyle = ParticleStyle.AURORA,
            voicePitch = 1.3f,
            glowIntensity = 1.1f,
        ),
        SomaAvatar(
            id = "void_stillness",
            name = "Void Stillness",
            description = "Near-invisible. Sacred silence made visual.",
            primaryColor = Color.parseColor("#4A5558"),
            secondaryColor = Color.parseColor("#2A3538"),
            fractalSeed = -0.4f,
            particleStyle = ParticleStyle.MINIMAL,
            voicePitch = 0.5f,
            glowIntensity = 0.5f,
        ),
        SomaAvatar(
            id = "aurora_spectrum",
            name = "Aurora Spectrum",
            description = "Shifting colors. Each harmony has its own hue.",
            primaryColor = Color.parseColor("#47D4FF"),
            secondaryColor = Color.parseColor("#6BCB77"),
            fractalSeed = 0.25f,
            particleStyle = ParticleStyle.AURORA,
            voicePitch = 1.1f,
            glowIntensity = 1.3f,
        ),
    )

    fun byId(id: String): SomaAvatar = avatars.find { it.id == id } ?: avatars[0]

    /** Load saved avatar from SharedPreferences. */
    fun load(context: Context): SomaAvatar {
        val prefs = context.getSharedPreferences("soma_prefs", Context.MODE_PRIVATE)
        val id = prefs.getString("avatar_id", "teal_consciousness") ?: "teal_consciousness"
        return byId(id)
    }

    /** Save selected avatar to SharedPreferences. */
    fun save(context: Context, avatar: SomaAvatar) {
        context.getSharedPreferences("soma_prefs", Context.MODE_PRIVATE)
            .edit().putString("avatar_id", avatar.id).apply()
    }
}
