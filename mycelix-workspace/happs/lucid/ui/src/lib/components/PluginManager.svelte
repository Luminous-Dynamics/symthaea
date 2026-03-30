<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<!--
  Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
  SPDX-License-Identifier: AGPL-3.0-or-later
  Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
-->
<script lang="ts">
	import {
		pluginRegistry,
		enablePlugin,
		disablePlugin,
		configurePlugin,
		unregisterPlugin,
		loadPluginFromURL,
		type Plugin,
		type PluginManifest,
		type PluginConfigSchema,
	} from '$services/plugin-system';

	let showAddPlugin = $state(false);
	let pluginUrl = $state('');
	let loadingPlugin = $state(false);
	let selectedPlugin: string | null = $state(null);

	function getPluginIcon(type: string): string {
		const icons: Record<string, string> = {
			importer: '📥',
			exporter: '📤',
			visualization: '📊',
			'ai-model': '🤖',
			integration: '🔗',
			theme: '🎨',
			command: '⌨️',
		};
		return icons[type] || '🔌';
	}

	function isEnabled(pluginId: string): boolean {
		return $pluginRegistry.enabled.has(pluginId);
	}

	function togglePlugin(pluginId: string) {
		if (isEnabled(pluginId)) {
			disablePlugin(pluginId);
		} else {
			enablePlugin(pluginId);
		}
	}

	async function handleLoadPlugin() {
		if (!pluginUrl.trim()) return;

		loadingPlugin = true;
		try {
			const plugin = await loadPluginFromURL(pluginUrl);
			if (plugin) {
				pluginUrl = '';
				showAddPlugin = false;
			}
		} finally {
			loadingPlugin = false;
		}
	}

	function handleRemovePlugin(pluginId: string) {
		if (confirm('Remove this plugin?')) {
			unregisterPlugin(pluginId);
		}
	}

	function getPluginConfig(pluginId: string): Record<string, any> {
		return $pluginRegistry.configs.get(pluginId) || {};
	}
</script>

<div class="plugin-manager">
	<div class="header">
		<h3>Plugins</h3>
		<button class="add-btn" onclick={() => (showAddPlugin = !showAddPlugin)}>
			{showAddPlugin ? '×' : '+'}
		</button>
	</div>

	{#if showAddPlugin}
		<div class="add-plugin-form">
			<input
				type="text"
				bind:value={pluginUrl}
				placeholder="Plugin URL or paste code..."
				class="plugin-url-input"
			/>
			<button
				class="load-btn"
				onclick={handleLoadPlugin}
				disabled={loadingPlugin || !pluginUrl.trim()}
			>
				{loadingPlugin ? 'Loading...' : 'Load'}
			</button>
		</div>
	{/if}

	<div class="plugin-list">
		{#each Array.from($pluginRegistry.plugins.entries()) as [id, plugin]}
			{@const manifest = plugin.manifest}
			{@const enabled = isEnabled(id)}

			<div class="plugin-item" class:enabled>
				<div class="plugin-header">
					<span class="plugin-icon">{getPluginIcon(manifest.type)}</span>
					<div class="plugin-info">
						<span class="plugin-name">{manifest.name}</span>
						<span class="plugin-version">v{manifest.version}</span>
					</div>
					<label class="toggle">
						<input type="checkbox" checked={enabled} onchange={() => togglePlugin(id)} />
						<span class="slider"></span>
					</label>
				</div>

				<p class="plugin-description">{manifest.description}</p>

				<div class="plugin-meta">
					<span class="plugin-type">{manifest.type}</span>
					<span class="plugin-author">by {manifest.author}</span>
				</div>

				{#if enabled && selectedPlugin === id}
					<div class="plugin-config">
						{#if manifest.config?.length}
							{#each manifest.config as configItem}
								{@const inputId = `plugin-${id}-${configItem.key}`}
								<div class="config-item">
									<label for={inputId}>{configItem.label}</label>
									{#if configItem.type === 'string'}
										<input
											id={inputId}
											type="text"
											value={getPluginConfig(id)[configItem.key] || configItem.default || ''}
											onchange={(e) => {
												const config = getPluginConfig(id);
												config[configItem.key] = e.currentTarget.value;
												configurePlugin(id, config);
											}}
										/>
									{:else if configItem.type === 'boolean'}
										<input
											id={inputId}
											type="checkbox"
											checked={getPluginConfig(id)[configItem.key] ?? configItem.default}
											onchange={(e) => {
												const config = getPluginConfig(id);
												config[configItem.key] = e.currentTarget.checked;
												configurePlugin(id, config);
											}}
										/>
									{:else if configItem.type === 'select' && configItem.options}
										<select
											id={inputId}
											value={getPluginConfig(id)[configItem.key] || configItem.default}
											onchange={(e) => {
												const config = getPluginConfig(id);
												config[configItem.key] = e.currentTarget.value;
												configurePlugin(id, config);
											}}
										>
											{#each configItem.options as opt}
												<option value={opt.value}>{opt.label}</option>
											{/each}
										</select>
									{/if}
								</div>
							{/each}
						{:else}
							<p class="no-config">No configuration options</p>
						{/if}
					</div>
				{/if}

				<div class="plugin-actions">
					{#if enabled && manifest.config?.length}
						<button
							class="config-btn"
							onclick={() => (selectedPlugin = selectedPlugin === id ? null : id)}
						>
							{selectedPlugin === id ? 'Hide Config' : 'Configure'}
						</button>
					{/if}
					{#if !id.startsWith('builtin-')}
						<button class="remove-btn" onclick={() => handleRemovePlugin(id)}>Remove</button>
					{/if}
				</div>
			</div>
		{/each}
	</div>

	{#if $pluginRegistry.plugins.size === 0}
		<div class="empty-state">
			<p>No plugins installed</p>
			<p class="hint">Add plugins to extend LUCID's capabilities</p>
		</div>
	{/if}
</div>

<style>
	.plugin-manager {
		padding-top: 24px;
	}

	.header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 16px;
	}

	h3 {
		margin: 0;
		font-size: 1rem;
		color: #e5e5e5;
	}

	.add-btn {
		width: 28px;
		height: 28px;
		background: #252540;
		border: 1px solid #3a3a5e;
		border-radius: 6px;
		color: #888;
		font-size: 1.2rem;
		cursor: pointer;
	}

	.add-btn:hover {
		color: #fff;
		border-color: #7c3aed;
	}

	.add-plugin-form {
		display: flex;
		gap: 8px;
		margin-bottom: 16px;
	}

	.plugin-url-input {
		flex: 1;
		padding: 8px 12px;
		background: #252540;
		border: 1px solid #3a3a5e;
		border-radius: 6px;
		color: #e5e5e5;
		font-size: 0.85rem;
	}

	.load-btn {
		padding: 8px 16px;
		background: #7c3aed;
		border: none;
		border-radius: 6px;
		color: white;
		font-size: 0.85rem;
		cursor: pointer;
	}

	.load-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.plugin-list {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.plugin-item {
		background: #1e1e2e;
		border: 1px solid #2a2a4e;
		border-radius: 8px;
		padding: 12px;
		transition: border-color 0.2s;
	}

	.plugin-item.enabled {
		border-color: #3a3a5e;
	}

	.plugin-header {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.plugin-icon {
		font-size: 1.5rem;
	}

	.plugin-info {
		flex: 1;
		display: flex;
		align-items: baseline;
		gap: 6px;
	}

	.plugin-name {
		font-weight: 600;
		color: #e5e5e5;
	}

	.plugin-version {
		font-size: 0.75rem;
		color: #666;
	}

	.toggle {
		position: relative;
		display: inline-block;
		width: 40px;
		height: 22px;
	}

	.toggle input {
		opacity: 0;
		width: 0;
		height: 0;
	}

	.slider {
		position: absolute;
		cursor: pointer;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background-color: #3a3a5e;
		transition: 0.2s;
		border-radius: 22px;
	}

	.slider:before {
		position: absolute;
		content: '';
		height: 16px;
		width: 16px;
		left: 3px;
		bottom: 3px;
		background-color: white;
		transition: 0.2s;
		border-radius: 50%;
	}

	input:checked + .slider {
		background-color: #7c3aed;
	}

	input:checked + .slider:before {
		transform: translateX(18px);
	}

	.plugin-description {
		margin: 8px 0;
		font-size: 0.85rem;
		color: #888;
		line-height: 1.4;
	}

	.plugin-meta {
		display: flex;
		gap: 12px;
		font-size: 0.75rem;
		color: #666;
	}

	.plugin-type {
		background: #252540;
		padding: 2px 8px;
		border-radius: 4px;
	}

	.plugin-config {
		margin-top: 12px;
		padding-top: 12px;
		border-top: 1px solid #2a2a4e;
	}

	.config-item {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 8px;
	}

	.config-item label {
		font-size: 0.85rem;
		color: #aaa;
	}

	.config-item input[type='text'],
	.config-item select {
		width: 150px;
		padding: 6px 10px;
		background: #252540;
		border: 1px solid #3a3a5e;
		border-radius: 4px;
		color: #e5e5e5;
		font-size: 0.85rem;
	}

	.no-config {
		color: #666;
		font-size: 0.85rem;
		font-style: italic;
	}

	.plugin-actions {
		display: flex;
		gap: 8px;
		margin-top: 12px;
	}

	.config-btn,
	.remove-btn {
		padding: 6px 12px;
		background: transparent;
		border: 1px solid #3a3a5e;
		border-radius: 4px;
		color: #888;
		font-size: 0.75rem;
		cursor: pointer;
	}

	.config-btn:hover {
		color: #e5e5e5;
		border-color: #7c3aed;
	}

	.remove-btn:hover {
		color: #ef4444;
		border-color: #ef4444;
	}

	.empty-state {
		text-align: center;
		padding: 32px;
		color: #666;
	}

	.empty-state .hint {
		font-size: 0.85rem;
		margin-top: 4px;
	}
</style>
