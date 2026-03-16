<script lang="ts">
  import { onMount } from 'svelte';
  import {
    getMyHearths,
    getEmergencyPlan,
    getHearthInventory,
    createEmergencyPlan,
    registerResource,
    raiseHearthAlert,
    type Hearth,
    type EmergencyPlan,
    type SharedResource,
    type EmergencyContact,
    type HearthResourceType,
    type HearthAlertType,
    type HearthAlertSeverity,
  } from '$lib/resilience-client';
  import { toasts } from '$lib/toast';

  let hearths: Hearth[] = [];
  let selectedHearth: Hearth | null = null;
  let plan: EmergencyPlan | null = null;
  let resources: SharedResource[] = [];
  let loading = true;
  let error = '';
  let submitting = false;

  // Emergency Plan form
  let showPlanForm = false;
  let planContacts: EmergencyContact[] = [{ name: '', relationship: '', phone: '', email: '' }];
  let planMeetingPoints = '';

  // Resource form
  let showResourceForm = false;
  let resName = '';
  let resDescription = '';
  let resType: HearthResourceType = 'Tools';
  let resCondition = 'Good';
  let resLocation = '';

  // Alert form
  let showAlertForm = false;
  let alertType: HearthAlertType = 'Other';
  let alertSeverity: HearthAlertSeverity = 'Medium';
  let alertMessage = '';

  const resourceTypes: HearthResourceType[] = ['Tools', 'Food', 'Water', 'Medical', 'Shelter', 'Communications'];
  const conditions = ['Excellent', 'Good', 'Fair', 'Poor'];
  const alertTypes: HearthAlertType[] = ['Fire', 'Flood', 'Medical', 'Violence', 'Other'];
  const alertSeverities: HearthAlertSeverity[] = ['Low', 'Medium', 'High', 'Critical'];

  onMount(async () => {
    try {
      hearths = await getMyHearths();
      if (hearths.length > 0) {
        await selectHearth(hearths[0]);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load household data';
    } finally {
      loading = false;
    }
  });

  async function selectHearth(hearth: Hearth) {
    selectedHearth = hearth;
    [plan, resources] = await Promise.all([
      getEmergencyPlan(hearth.id),
      getHearthInventory(hearth.id),
    ]);
  }

  function addContact() {
    planContacts = [...planContacts, { name: '', relationship: '', phone: '', email: '' }];
  }

  function removeContact(index: number) {
    const name = planContacts[index]?.name?.trim() || 'this contact';
    if (!confirm(`Remove ${name} from emergency contacts?`)) return;
    planContacts = planContacts.filter((_, i) => i !== index);
  }

  async function handleCreatePlan() {
    if (!selectedHearth) return;
    const validContacts = planContacts.filter(c => c.name.trim() && c.phone.trim());
    if (validContacts.length === 0) return;
    const points = planMeetingPoints.split('\n').map(p => p.trim()).filter(p => p.length > 0);
    submitting = true;
    try {
      plan = await createEmergencyPlan(selectedHearth.id, validContacts, points);
      planContacts = [{ name: '', relationship: '', phone: '', email: '' }];
      planMeetingPoints = '';
      showPlanForm = false;
      toasts.success('Plan saved');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to save plan');
    } finally {
      submitting = false;
    }
  }

  async function handleRegisterResource() {
    if (!selectedHearth || !resName.trim()) return;
    submitting = true;
    try {
      const res = await registerResource(
        selectedHearth.id,
        resName.trim(),
        resDescription.trim(),
        resType,
        resCondition,
        resLocation.trim(),
      );
      resources = [...resources, res];
      resName = '';
      resDescription = '';
      resType = 'Tools';
      resCondition = 'Good';
      resLocation = '';
      showResourceForm = false;
      toasts.success('Resource shared');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to share resource');
    } finally {
      submitting = false;
    }
  }

  async function handleRaiseAlert() {
    if (!selectedHearth || !alertMessage.trim()) return;
    if (!confirm('This will alert all household members. Continue?')) return;
    submitting = true;
    try {
      await raiseHearthAlert(selectedHearth.id, alertType, alertSeverity, alertMessage.trim());
      alertType = 'Other';
      alertSeverity = 'Medium';
      alertMessage = '';
      showAlertForm = false;
      toasts.success('Alert raised');
    } catch (e) {
      toasts.error(e instanceof Error ? e.message : 'Failed to raise alert');
    } finally {
      submitting = false;
    }
  }

  function daysSince(timestamp: number): number {
    return Math.floor((Date.now() - timestamp) / 86400000);
  }

  function reviewStatus(plan: EmergencyPlan): { label: string; color: string } {
    const days = daysSince(plan.last_reviewed);
    if (days < 30) return { label: 'Current', color: 'text-green-400' };
    if (days < 90) return { label: `${days}d ago`, color: 'text-yellow-400' };
    return { label: `${days}d ago — review needed`, color: 'text-red-400' };
  }

  function severityColor(severity: string): string {
    switch (severity) {
      case 'Critical': return 'bg-red-600 text-white';
      case 'High': return 'bg-orange-500/20 text-orange-400 border border-orange-500/50';
      case 'Medium': return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50';
      case 'Low': return 'bg-gray-600 text-gray-300';
      default: return 'bg-gray-600 text-gray-300';
    }
  }

  const resourceTypeIcons: Record<string, string> = {
    Tools: '🔧',
    Food: '🍞',
    Water: '💧',
    Medical: '🏥',
    Shelter: '🏠',
    Communications: '📡',
  };
</script>

<div class="min-h-screen bg-gray-950 text-gray-100 p-6">
  <div class="container mx-auto max-w-6xl">
    <h1 class="text-2xl font-bold mb-1">Household Emergency</h1>
    <p class="text-gray-400 mb-6">Emergency plans, shared resources, and family coordination.</p>

    {#if loading}
      <div class="text-gray-400">Loading household data...</div>
    {:else if error}
      <div class="bg-red-900/30 border border-red-500 rounded p-4 text-red-200">{error}</div>
    {:else}

      <!-- Hearth Selector -->
      {#if hearths.length > 1}
        <div class="flex flex-wrap gap-2 mb-6">
          {#each hearths as hearth}
            <button
              class="px-4 py-2 rounded-lg border transition-colors {selectedHearth?.id === hearth.id ? 'bg-indigo-900/50 border-indigo-500 text-indigo-200' : 'bg-gray-900 border-gray-700 text-gray-400 hover:border-gray-600'}"
              on:click={() => selectHearth(hearth)}
              aria-label="Select household {hearth.name}"
            >
              {hearth.name}
              <span class="text-xs ml-1 opacity-75">({hearth.member_count} members)</span>
            </button>
          {/each}
        </div>
      {/if}

      {#if selectedHearth}

        <!-- Action Buttons -->
        <div class="flex gap-3 mb-6">
          <button on:click={() => { showPlanForm = !showPlanForm; showResourceForm = false; showAlertForm = false; }}
            class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm font-medium transition-colors"
            aria-label="Create a new emergency plan">
            + Emergency Plan
          </button>
          <button on:click={() => { showResourceForm = !showResourceForm; showPlanForm = false; showAlertForm = false; }}
            class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm font-medium transition-colors"
            aria-label="Share a resource with your household">
            + Share Resource
          </button>
          <button on:click={() => { showAlertForm = !showAlertForm; showPlanForm = false; showResourceForm = false; }}
            class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
            aria-label="Raise an emergency alert">
            Raise Alert
          </button>
        </div>

        <!-- Emergency Plan Form -->
        {#if showPlanForm}
          <div class="bg-gray-900 rounded-lg border border-gray-800 p-6 mb-6">
            <h2 class="text-sm font-semibold text-gray-300 mb-4">Create Emergency Plan</h2>
            <form on:submit|preventDefault={handleCreatePlan} class="space-y-4">
              <!-- Contacts -->
              <div>
                <div class="flex items-center justify-between mb-2">
                  <span class="text-xs text-gray-400 uppercase tracking-wider">Emergency Contacts</span>
                  <button type="button" on:click={addContact}
                    class="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded transition-colors">
                    + Add Contact
                  </button>
                </div>
                <div class="space-y-3">
                  {#each planContacts as contact, i}
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2 items-end bg-gray-800/50 rounded p-3">
                      <div>
                        <label for="cname-{i}" class="text-xs text-gray-500">Name</label>
                        <input id="cname-{i}" bind:value={contact.name} placeholder="Jane Doe"
                          class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
                      </div>
                      <div>
                        <label for="crel-{i}" class="text-xs text-gray-500">Relationship</label>
                        <input id="crel-{i}" bind:value={contact.relationship} placeholder="Neighbor"
                          class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
                      </div>
                      <div>
                        <label for="cphone-{i}" class="text-xs text-gray-500">Phone</label>
                        <input id="cphone-{i}" bind:value={contact.phone} placeholder="+27 82 000 0000"
                          class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
                      </div>
                      <div>
                        <label for="cemail-{i}" class="text-xs text-gray-500">Email</label>
                        <input id="cemail-{i}" bind:value={contact.email} placeholder="jane@example.com"
                          class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
                      </div>
                      <div>
                        {#if planContacts.length > 1}
                          <button type="button" on:click={() => removeContact(i)}
                            class="w-full mt-1 bg-red-900/50 hover:bg-red-900 border border-red-700 text-red-300 rounded px-3 py-2 text-sm transition-colors"
                            aria-label="Remove contact {contact.name || 'unnamed'}">
                            Remove
                          </button>
                        {/if}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>

              <!-- Meeting Points -->
              <div>
                <label for="mpoints" class="text-xs text-gray-400 uppercase tracking-wider">Meeting Points (one per line)</label>
                <textarea id="mpoints" bind:value={planMeetingPoints} rows="3"
                  placeholder="Front gate of property&#10;Community hall on 5th Ave&#10;Park pavilion near the dam"
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"></textarea>
              </div>

              <button type="submit"
                disabled={submitting || planContacts.every(c => !c.name.trim() || !c.phone.trim())}
                class="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors"
                aria-label="Submit emergency plan">
                {submitting ? 'Creating...' : 'Create Emergency Plan'}
              </button>
            </form>
          </div>
        {/if}

        <!-- Share Resource Form -->
        {#if showResourceForm}
          <div class="bg-gray-900 rounded-lg border border-gray-800 p-6 mb-6">
            <h2 class="text-sm font-semibold text-gray-300 mb-4">Share a Resource</h2>
            <form on:submit|preventDefault={handleRegisterResource} class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label for="rname" class="text-xs text-gray-400">Resource Name</label>
                <input id="rname" bind:value={resName} placeholder="Generator, first aid kit, etc."
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
              </div>
              <div>
                <label for="rtype" class="text-xs text-gray-400">Type</label>
                <select id="rtype" bind:value={resType}
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                  {#each resourceTypes as t}
                    <option value={t}>{resourceTypeIcons[t] ?? '📦'} {t}</option>
                  {/each}
                </select>
              </div>
              <div class="md:col-span-2">
                <label for="rdesc" class="text-xs text-gray-400">Description</label>
                <input id="rdesc" bind:value={resDescription} placeholder="Brief description of the resource"
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
              </div>
              <div>
                <label for="rcond" class="text-xs text-gray-400">Condition</label>
                <select id="rcond" bind:value={resCondition}
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                  {#each conditions as c}
                    <option value={c}>{c}</option>
                  {/each}
                </select>
              </div>
              <div>
                <label for="rloc" class="text-xs text-gray-400">Location</label>
                <input id="rloc" bind:value={resLocation} placeholder="Garage, shed, etc."
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-indigo-500" />
              </div>
              <div class="md:col-span-2">
                <button type="submit" disabled={submitting || !resName.trim()}
                  class="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors"
                  aria-label="Submit shared resource">
                  {submitting ? 'Registering...' : 'Register Resource'}
                </button>
              </div>
            </form>
          </div>
        {/if}

        <!-- Raise Alert Form -->
        {#if showAlertForm}
          <div class="bg-gray-900 rounded-lg border border-red-800/50 p-6 mb-6">
            <h2 class="text-sm font-semibold text-red-300 mb-4">Raise Hearth Alert</h2>
            <form on:submit|preventDefault={handleRaiseAlert} class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label for="atype" class="text-xs text-gray-400">Alert Type</label>
                <select id="atype" bind:value={alertType}
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-red-500">
                  {#each alertTypes as t}
                    <option value={t}>{t}</option>
                  {/each}
                </select>
              </div>
              <div>
                <label for="asev" class="text-xs text-gray-400">Severity</label>
                <select id="asev" bind:value={alertSeverity}
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-red-500">
                  {#each alertSeverities as s}
                    <option value={s}>{s}</option>
                  {/each}
                </select>
              </div>
              <div class="md:col-span-2">
                <label for="amsg" class="text-xs text-gray-400">Message</label>
                <textarea id="amsg" bind:value={alertMessage} rows="3"
                  placeholder="Describe the emergency situation..."
                  class="w-full mt-1 bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none focus:border-red-500"></textarea>
              </div>
              <div class="md:col-span-2 flex items-center gap-3">
                <span class={`text-xs px-2 py-1 rounded ${severityColor(alertSeverity)}`}>{alertSeverity}</span>
                <span class="text-xs text-gray-500">This alert will be sent to all hearth members.</span>
              </div>
              <div class="md:col-span-2">
                <button type="submit" disabled={submitting || !alertMessage.trim()}
                  class="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors"
                  aria-label="Send emergency alert to all household members">
                  {submitting ? 'Sending Alert...' : 'Send Alert'}
                </button>
              </div>
            </form>
          </div>
        {/if}

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

          <!-- Emergency Plan -->
          <div class="bg-gray-900 rounded-lg border border-gray-800 p-5">
            <h2 class="text-lg font-semibold mb-4">Emergency Plan</h2>
            {#if plan}
              {@const status = reviewStatus(plan)}
              <div class="mb-4">
                <div class="flex items-center justify-between mb-3">
                  <span class="text-sm text-gray-400">Last reviewed</span>
                  <span class="text-sm font-medium {status.color}">{status.label}</span>
                </div>

                <h3 class="text-sm font-semibold text-gray-300 mb-2 uppercase tracking-wider">Emergency Contacts</h3>
                <div class="space-y-2 mb-4">
                  {#each plan.contacts as contact}
                    <div class="flex items-center justify-between bg-gray-800/50 rounded px-3 py-2">
                      <div>
                        <span class="font-medium text-white">{contact.name}</span>
                        <span class="text-xs text-gray-400 ml-2">{contact.relationship}</span>
                      </div>
                      <a href="tel:{contact.phone}" class="text-cyan-400 hover:text-cyan-300 text-sm font-mono" aria-label="Call {contact.name} at {contact.phone}">{contact.phone}</a>
                    </div>
                  {/each}
                </div>

                <h3 class="text-sm font-semibold text-gray-300 mb-2 uppercase tracking-wider">Meeting Points</h3>
                <ol class="space-y-1">
                  {#each plan.meeting_points as point, i}
                    <li class="flex items-start gap-2 text-sm">
                      <span class="inline-flex items-center justify-center w-5 h-5 rounded-full bg-gray-700 text-gray-300 text-xs font-bold flex-shrink-0">{i + 1}</span>
                      <span class="text-gray-300">{point}</span>
                    </li>
                  {/each}
                </ol>
              </div>
            {:else}
              <div class="text-center text-gray-500 py-8">
                <p>No emergency plan created yet.</p>
                <p class="text-sm mt-1">Create a plan with emergency contacts and meeting points.</p>
              </div>
            {/if}
          </div>

          <!-- Shared Resources -->
          <div class="bg-gray-900 rounded-lg border border-gray-800 p-5">
            <div class="flex items-center justify-between mb-4">
              <h2 class="text-lg font-semibold">Shared Resources</h2>
              <span class="text-sm text-gray-400">{resources.length} items</span>
            </div>

            {#if resources.length > 0}
              <div class="space-y-2">
                {#each resources as resource}
                  <div class="bg-gray-800/50 rounded-lg px-4 py-3">
                    <div class="flex items-start justify-between">
                      <div>
                        <span class="mr-1">{resourceTypeIcons[resource.resource_type] ?? '📦'}</span>
                        <span class="font-medium text-white">{resource.name}</span>
                      </div>
                      <span class="text-xs px-2 py-0.5 rounded-full {resource.current_holder ? 'bg-yellow-900/50 text-yellow-300' : 'bg-green-900/50 text-green-300'}">
                        {resource.current_holder ? 'Lent out' : 'Available'}
                      </span>
                    </div>
                    <p class="text-sm text-gray-400 mt-1">{resource.description}</p>
                    <div class="flex gap-4 mt-1 text-xs text-gray-500">
                      <span>Condition: {resource.condition}</span>
                      <span>Location: {resource.location}</span>
                    </div>
                    {#if resource.current_holder}
                      <p class="text-xs text-yellow-400 mt-1">Currently with: {resource.current_holder}</p>
                    {/if}
                  </div>
                {/each}
              </div>
            {:else}
              <div class="text-center text-gray-500 py-8">
                <p>No shared resources registered.</p>
                <p class="text-sm mt-1">Register tools, medical supplies, and emergency equipment your household can share.</p>
              </div>
            {/if}
          </div>

        </div>
      {:else}
        <div class="text-center text-gray-500 py-12">
          <p class="text-lg">No households found.</p>
          <p class="text-sm mt-1">Create a hearth to start coordinating emergency preparedness with your family and neighbors.</p>
        </div>
      {/if}
    {/if}
  </div>
</div>
