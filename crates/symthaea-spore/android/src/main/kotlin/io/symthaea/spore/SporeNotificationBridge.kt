package io.symthaea.spore

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

/**
 * Bridges notification count to the Spore engine as social pressure.
 *
 * Requires user to explicitly enable in Settings → Apps → Special access →
 * Notification access. No notification content is read — only count.
 *
 * High count (>5) → mild NE nudge (social demand/pressure).
 * Zero for extended period → oxytocin decay (isolation signal).
 */
class SporeNotificationBridge : NotificationListenerService() {

    override fun onNotificationPosted(sbn: StatusBarNotification?) {
        updateCount()
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification?) {
        updateCount()
    }

    private fun updateCount() {
        val count = try {
            activeNotifications?.size ?: 0
        } catch (_: Exception) {
            0
        }
        engineRef?.setSocialPressure(count)
    }

    companion object {
        /** Set by the activity/service that owns the SporeEngine. */
        @Volatile
        var engineRef: SporeEngine? = null
    }
}
