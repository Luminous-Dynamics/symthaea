<script lang="ts">
  import { locale, setLocale, type Locale } from '$lib/i18n';

  const locales: { code: Locale; label: string }[] = [
    { code: 'en', label: 'English' },
    { code: 'af', label: 'Afrikaans' },
    { code: 'zu', label: 'isiZulu' },
    { code: 'st', label: 'Sesotho' },
  ];

  function handleChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    setLocale(target.value as Locale);
  }
</script>

<select
  value={$locale}
  on:change={handleChange}
  aria-label="Select language"
  class="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300 focus:outline-none focus:border-indigo-500"
>
  {#each locales as loc}
    <option value={loc.code} disabled={loc.code !== 'en'}>{loc.label}{loc.code !== 'en' ? ' (v0.2)' : ''}</option>
  {/each}
</select>
