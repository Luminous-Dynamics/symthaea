package io.symthaea.soma

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.Closeable

/**
 * Coroutine-friendly wrapper around [SomaEngine].
 *
 * Dispatches all native calls to a dedicated single thread,
 * preventing concurrent access to the non-thread-safe engine.
 *
 * Uses [SupervisorJob] so that a failure in one coroutine (e.g., a dream
 * cycle crash) does not cancel sibling coroutines or the entire scope.
 *
 * Lifecycle:
 * - [start] creates the scope and dispatcher (idempotent).
 * - [stop] cancels all running coroutines without closing the engine.
 * - [close] cancels, closes the engine, and closes the dispatcher.
 *
 * After [stop], [start] can be called again to resume operation (e.g.,
 * after an Activity is recreated). After [close], the instance is dead.
 */
class SomaCoroutineEngine(
    private val engine: SomaEngine,
    private val dispatcher: CloseableCoroutineDispatcher = @OptIn(ExperimentalCoroutinesApi::class) newSingleThreadContext("SomaEngine"),
) : Closeable {

    private val supervisorJob = SupervisorJob()
    private var scope = CoroutineScope(dispatcher + supervisorJob)

    /** Whether the scope is active and accepting new coroutines. */
    val isActive: Boolean get() = supervisorJob.isActive

    /** Run one consciousness cycle. */
    suspend fun cycle(input: String? = null): CycleResult =
        withContext(dispatcher) { engine.cycle(input) }

    /** Raw cycle returning just consciousness level (lower overhead). */
    suspend fun cycleRaw(input: String? = null): Float =
        withContext(dispatcher) { engine.cycleRaw(input) }

    /**
     * Continuous consciousness flow at the given interval.
     *
     * Default 50ms = 20Hz, matching the mobile-optimized engine target.
     * Collect this flow from a ViewModel to drive UI updates.
     *
     * Individual cycle failures are caught and logged so they don't kill the flow.
     */
    fun cycleFlow(
        intervalMs: Long = 50L,
        input: () -> String? = { null },
    ): Flow<CycleResult> = flow {
        while (currentCoroutineContext().isActive) {
            try {
                emit(engine.cycle(input()))
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                android.util.Log.e(TAG, "Cycle flow error", e)
            }
            delay(intervalMs)
        }
    }.flowOn(dispatcher)

    /** Compass snapshot flow for widget updates (lower frequency). */
    fun compassFlow(intervalMs: Long = 200L): Flow<CompassSnapshot> = flow {
        while (currentCoroutineContext().isActive) {
            try {
                emit(engine.compassSnapshot())
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                android.util.Log.e(TAG, "Compass flow error", e)
            }
            delay(intervalMs)
        }
    }.flowOn(dispatcher)

    /** Launch a coroutine in the supervised scope. Failures won't cancel siblings. */
    fun launch(block: suspend CoroutineScope.() -> Unit): Job {
        return scope.launch(block = block)
    }

    /** Run a dream cycle off the main thread. */
    suspend fun dreamCycle(): Boolean =
        withContext(dispatcher) { engine.dreamCycle() }

    /** Explicit dream consolidation. */
    suspend fun dreamConsolidate() =
        withContext(dispatcher) { engine.dreamConsolidate() }

    /**
     * Cancel all running coroutines. The engine remains open and the
     * dispatcher stays alive, so [start] can reinitialize the scope.
     */
    fun stop() {
        supervisorJob.cancel()
    }

    /**
     * Restart the coroutine scope after a [stop]. Creates a fresh
     * SupervisorJob so new coroutines can be launched.
     *
     * No-op if the scope is already active.
     */
    fun start() {
        if (supervisorJob.isActive) return
        // SupervisorJob cannot be reused after cancellation — but since the
        // field is val, we recreate the scope with a new child supervisor.
        val newJob = SupervisorJob()
        scope = CoroutineScope(dispatcher + newJob)
    }

    /** Access the underlying engine for direct (non-suspend) operations. */
    val raw: SomaEngine get() = engine

    /**
     * Cancel all coroutines, close the engine, and release the dispatcher.
     * After this call the instance is unusable.
     */
    override fun close() {
        supervisorJob.cancel()
        engine.close()
        dispatcher.close()
    }

    companion object {
        private const val TAG = "SomaCoroutine"
    }
}
