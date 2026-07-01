# Keep JNI native bindings (must not be obfuscated/removed)
-keep class io.symthaea.spore.NativeBindings { *; }

# Keep serializable data classes (kotlinx.serialization)
-keep class io.symthaea.spore.CycleResult { *; }
-keep class io.symthaea.spore.EpistemicStatus { *; }
-keep class io.symthaea.spore.CompassSnapshot { *; }
-keep class io.symthaea.spore.DreamFragment { *; }
-keep class io.symthaea.spore.SporeConfig { *; }
-keep class io.symthaea.spore.SharingConfig { *; }
