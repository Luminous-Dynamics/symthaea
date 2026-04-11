// Mycelix Pulse Android - Theme

package com.mycelix.mail.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// Brand Colors
val MycelixPrimary = Color(0xFF4A90D9)
val MycelixSecondary = Color(0xFF9463D9)
val MycelixTertiary = Color(0xFFD9A343)

// Light Theme Colors
private val LightColorScheme = lightColorScheme(
    primary = MycelixPrimary,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD6E3F7),
    onPrimaryContainer = Color(0xFF001B3D),

    secondary = MycelixSecondary,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFE8DEFF),
    onSecondaryContainer = Color(0xFF21005D),

    tertiary = MycelixTertiary,
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFFFF0D6),
    onTertiaryContainer = Color(0xFF3D2E00),

    error = Color(0xFFBA1A1A),
    onError = Color.White,
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF410002),

    background = Color(0xFFFCFCFC),
    onBackground = Color(0xFF1B1B1F),

    surface = Color.White,
    onSurface = Color(0xFF1B1B1F),
    surfaceVariant = Color(0xFFE7E0EC),
    onSurfaceVariant = Color(0xFF49454F),

    outline = Color(0xFF79747E),
    outlineVariant = Color(0xFFCAC4D0)
)

// Dark Theme Colors
private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFFAAC7FF),
    onPrimary = Color(0xFF002F64),
    primaryContainer = Color(0xFF00458D),
    onPrimaryContainer = Color(0xFFD6E3FF),

    secondary = Color(0xFFCFBDFF),
    onSecondary = Color(0xFF3B0094),
    secondaryContainer = Color(0xFF5200CE),
    onSecondaryContainer = Color(0xFFE8DEFF),

    tertiary = Color(0xFFFFD970),
    onTertiary = Color(0xFF3D2E00),
    tertiaryContainer = Color(0xFF584400),
    onTertiaryContainer = Color(0xFFFFE08E),

    error = Color(0xFFFFB4AB),
    onError = Color(0xFF690005),
    errorContainer = Color(0xFF93000A),
    onErrorContainer = Color(0xFFFFDAD6),

    background = Color(0xFF1B1B1F),
    onBackground = Color(0xFFE4E1E6),

    surface = Color(0xFF1B1B1F),
    onSurface = Color(0xFFE4E1E6),
    surfaceVariant = Color(0xFF49454F),
    onSurfaceVariant = Color(0xFFCAC4D0),

    outline = Color(0xFF938F99),
    outlineVariant = Color(0xFF49454F)
)

@Composable
fun MycelixMailTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.surface.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}

val Typography = Typography()
