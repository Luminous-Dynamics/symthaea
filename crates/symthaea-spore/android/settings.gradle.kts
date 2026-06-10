pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "symthaea-spore-android"

// Root directory is the :spore library module (source in src/main/kotlin/...)
// Demo app is a subproject
include(":demo")
