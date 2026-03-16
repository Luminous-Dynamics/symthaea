<script>
	import { page } from '$app/stores';
</script>

<div class="error-container">
	<h1 class="status-code">{$page.status}</h1>
	<h2 class="heading">Something went wrong</h2>
	<p class="message">{$page.error?.message ?? 'An unexpected error occurred.'}</p>
	<div class="actions">
		<a href="/resilience" class="btn btn-primary">Back to Dashboard</a>
		<button class="btn btn-secondary" on:click={() => location.reload()}>Try again</button>
	</div>
</div>

<style>
	.error-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 100vh;
		background-color: rgb(3 7 18); /* bg-gray-950 */
		color: rgb(243 244 246); /* text-gray-100 */
		padding: 2rem;
		text-align: center;
	}

	.status-code {
		font-size: 5rem;
		font-weight: 700;
		line-height: 1;
		color: rgb(79 70 229); /* indigo-600 */
		margin: 0 0 0.5rem;
	}

	.heading {
		font-size: 1.5rem;
		font-weight: 600;
		margin: 0 0 1rem;
	}

	.message {
		color: rgb(156 163 175); /* text-gray-400 */
		font-size: 1rem;
		margin: 0 0 2rem;
		max-width: 28rem;
	}

	.actions {
		display: flex;
		gap: 1rem;
	}

	.btn {
		padding: 0.625rem 1.25rem;
		border-radius: 0.5rem;
		font-size: 0.875rem;
		font-weight: 500;
		cursor: pointer;
		text-decoration: none;
		transition: background-color 150ms ease;
	}

	.btn-primary {
		background-color: rgb(79 70 229); /* bg-indigo-600 */
		color: rgb(243 244 246);
		border: none;
	}

	.btn-primary:hover {
		background-color: rgb(67 56 202); /* bg-indigo-700 */
	}

	.btn-secondary {
		background-color: transparent;
		color: rgb(243 244 246);
		border: 1px solid rgb(75 85 99); /* border-gray-600 */
	}

	.btn-secondary:hover {
		background-color: rgb(31 41 55); /* bg-gray-800 */
	}
</style>
