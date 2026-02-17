<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  // Redirect to browse on homepage load
  onMount(() => {
    goto('/browse');
  });
</script>

<main class="homepage">
  <div class="container">
    <div class="hero">
      <h1>Mycelix Marketplace</h1>
      <p>Loading...</p>
    </div>
  </div>
</main>

<style>
  .homepage {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }

  .hero {
    text-align: center;
    color: white;
  }

  h1 {
    font-size: 3rem;
    margin-bottom: 1rem;
    color: white;
  }

  p {
    font-size: 1.2rem;
    opacity: 0.9;
  }
</style>
