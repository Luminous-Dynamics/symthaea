# Keep JNI native bindings (must not be obfuscated/removed)
-keep class io.symthaea.soma.NativeBindings { *; }

# Keep serializable data classes (kotlinx.serialization)
-keep class io.symthaea.soma.CycleResult { *; }
-keep class io.symthaea.soma.EpistemicStatus { *; }
-keep class io.symthaea.soma.CompassSnapshot { *; }
-keep class io.symthaea.soma.DreamFragment { *; }
-keep class io.symthaea.soma.SomaConfig { *; }
-keep class io.symthaea.soma.SharingConfig { *; }
