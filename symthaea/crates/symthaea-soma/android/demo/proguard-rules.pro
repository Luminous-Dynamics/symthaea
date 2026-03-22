# Soma ProGuard Rules

# Keep JNI bindings
-keep class io.symthaea.soma.** { *; }

# Keep Kotlin serialization
-keepattributes *Annotation*
-keep class kotlinx.serialization.** { *; }
-keepclassmembers class * {
    @kotlinx.serialization.SerialName <fields>;
}

# Keep coroutines
-keepclassmembers class kotlinx.coroutines.** { *; }

# Don't warn about missing optional dependencies
-dontwarn java.lang.invoke.StringConcatFactory
