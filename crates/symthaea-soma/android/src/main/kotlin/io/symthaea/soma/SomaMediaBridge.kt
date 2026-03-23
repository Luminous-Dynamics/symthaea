package io.symthaea.soma

import android.content.ComponentName
import android.content.Context
import android.media.session.MediaController
import android.media.session.MediaSessionManager
import android.media.session.PlaybackState
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner

/**
 * Bridges media playback state to the Soma engine.
 *
 * Music playing → serotonin nudge (relaxation/enjoyment).
 * Podcast/speech → dopamine nudge (learning/curiosity).
 * No content access — just playback state (playing/paused/stopped).
 *
 * Requires notification listener permission (shares with [SomaNotificationBridge]).
 *
 * Usage:
 * ```kotlin
 * lifecycle.addObserver(SomaMediaBridge(context, engine))
 * ```
 */
class SomaMediaBridge(
    private val context: Context,
    private val engine: SomaEngine,
) : DefaultLifecycleObserver {

    private var sessionManager: MediaSessionManager? = null

    private val sessionListener =
        MediaSessionManager.OnActiveSessionsChangedListener { controllers ->
            updateMediaState(controllers)
        }

    override fun onStart(owner: LifecycleOwner) {
        try {
            sessionManager =
                context.getSystemService(Context.MEDIA_SESSION_SERVICE) as? MediaSessionManager
            val component = ComponentName(context, SomaNotificationBridge::class.java)
            sessionManager?.addOnActiveSessionsChangedListener(sessionListener, component)
            // Initial check
            updateMediaState(sessionManager?.getActiveSessions(component))
        } catch (_: SecurityException) {
            // Notification listener not enabled — media bridge inactive
        }
    }

    override fun onStop(owner: LifecycleOwner) {
        sessionManager?.removeOnActiveSessionsChangedListener(sessionListener)
        sessionManager = null
    }

    private fun updateMediaState(controllers: List<MediaController>?) {
        if (controllers.isNullOrEmpty()) {
            engine.setMediaState(MEDIA_NONE)
            return
        }
        // Check the first active controller
        val state = controllers[0].playbackState?.state
        if (state == PlaybackState.STATE_PLAYING) {
            // Heuristic: if audio info suggests speech content, use SPEECH
            // For now, default to MUSIC (most common media playback)
            engine.setMediaState(MEDIA_MUSIC)
        } else {
            engine.setMediaState(MEDIA_NONE)
        }
    }

    companion object {
        const val MEDIA_NONE = 0
        const val MEDIA_MUSIC = 1
        const val MEDIA_SPEECH = 2
    }
}
