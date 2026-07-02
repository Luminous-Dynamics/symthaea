// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Interactive Widgets for Mycelix Protocol Documentation
 * Provides hands-on learning experiences
 */

// ============================================================================
// Byzantine Tolerance Calculator
// ============================================================================

function createByzantineCalculator() {
    const container = document.getElementById('byzantine-calculator');
    if (!container) return;

    container.innerHTML = `
        <div class="interactive-widget">
            <h3>🛡️ Byzantine Tolerance Calculator</h3>
            <p>Explore how reputation affects Byzantine tolerance</p>

            <div class="widget-controls">
                <div class="control-group">
                    <label for="total-nodes">Total Nodes:</label>
                    <input type="range" id="total-nodes" min="10" max="100" value="20" step="1">
                    <span id="total-nodes-value">20</span>
                </div>

                <div class="control-group">
                    <label for="byzantine-ratio">Byzantine Ratio (%):</label>
                    <input type="range" id="byzantine-ratio" min="0" max="60" value="20" step="1">
                    <span id="byzantine-ratio-value">20%</span>
                </div>

                <div class="control-group">
                    <label for="byzantine-reputation">Byzantine Node Reputation:</label>
                    <input type="range" id="byzantine-reputation" min="0" max="1" value="0.1" step="0.05">
                    <span id="byzantine-reputation-value">0.10</span>
                </div>

                <div class="control-group">
                    <label for="honest-reputation">Honest Node Reputation:</label>
                    <input type="range" id="honest-reputation" min="0" max="1" value="0.9" step="0.05">
                    <span id="honest-reputation-value">0.90</span>
                </div>
            </div>

            <div class="widget-results">
                <div class="result-card" id="byzantine-power-card">
                    <h4>Byzantine Power</h4>
                    <div class="result-value" id="byzantine-power">0</div>
                </div>

                <div class="result-card" id="honest-power-card">
                    <h4>Honest Power</h4>
                    <div class="result-value" id="honest-power">0</div>
                </div>

                <div class="result-card" id="system-status-card">
                    <h4>System Status</h4>
                    <div class="result-value" id="system-status">SAFE</div>
                </div>

                <div class="result-card" id="tolerance-card">
                    <h4>Max Byzantine Tolerance</h4>
                    <div class="result-value" id="max-tolerance">45%</div>
                </div>
            </div>

            <div class="widget-visualization">
                <canvas id="byzantine-chart" width="600" height="300"></canvas>
            </div>

            <div class="widget-explanation">
                <p><strong>How it works:</strong></p>
                <p>Byzantine Power = Σ(malicious_reputation²)</p>
                <p>System is SAFE when: Byzantine_Power < Honest_Power / 3</p>
            </div>
        </div>
    `;

    const inputs = {
        totalNodes: document.getElementById('total-nodes'),
        byzantineRatio: document.getElementById('byzantine-ratio'),
        byzantineRep: document.getElementById('byzantine-reputation'),
        honestRep: document.getElementById('honest-reputation'),
    };

    const outputs = {
        totalNodesValue: document.getElementById('total-nodes-value'),
        byzantineRatioValue: document.getElementById('byzantine-ratio-value'),
        byzantineRepValue: document.getElementById('byzantine-reputation-value'),
        honestRepValue: document.getElementById('honest-reputation-value'),
        byzantinePower: document.getElementById('byzantine-power'),
        honestPower: document.getElementById('honest-power'),
        systemStatus: document.getElementById('system-status'),
        maxTolerance: document.getElementById('max-tolerance'),
    };

    function updateCalculation() {
        const totalNodes = parseInt(inputs.totalNodes.value);
        const byzantineRatio = parseInt(inputs.byzantineRatio.value) / 100;
        const byzantineRep = parseFloat(inputs.byzantineRep.value);
        const honestRep = parseFloat(inputs.honestRep.value);

        const byzantineNodes = Math.floor(totalNodes * byzantineRatio);
        const honestNodes = totalNodes - byzantineNodes;

        // Calculate powers (reputation-weighted)
        const byzantinePower = byzantineNodes * Math.pow(byzantineRep, 2);
        const honestPower = honestNodes * Math.pow(honestRep, 2);

        // System is safe when: Byzantine_Power < Honest_Power / 3
        const threshold = honestPower / 3;
        const isSafe = byzantinePower < threshold;

        // Calculate max tolerance (binary search)
        let maxTolerance = 0;
        for (let ratio = 0; ratio <= 100; ratio++) {
            const testByzNodes = Math.floor(totalNodes * (ratio / 100));
            const testHonestNodes = totalNodes - testByzNodes;
            const testByzPower = testByzNodes * Math.pow(byzantineRep, 2);
            const testHonestPower = testHonestNodes * Math.pow(honestRep, 2);

            if (testByzPower < testHonestPower / 3) {
                maxTolerance = ratio;
            } else {
                break;
            }
        }

        // Update outputs
        outputs.totalNodesValue.textContent = totalNodes;
        outputs.byzantineRatioValue.textContent = `${Math.round(byzantineRatio * 100)}%`;
        outputs.byzantineRepValue.textContent = byzantineRep.toFixed(2);
        outputs.honestRepValue.textContent = honestRep.toFixed(2);

        outputs.byzantinePower.textContent = byzantinePower.toFixed(2);
        outputs.honestPower.textContent = honestPower.toFixed(2);
        outputs.systemStatus.textContent = isSafe ? '✅ SAFE' : '❌ UNSAFE';
        outputs.systemStatus.style.color = isSafe ? '#00e676' : '#f44336';
        outputs.maxTolerance.textContent = `${maxTolerance}%`;

        // Update visualization
        updateChart(byzantineNodes, honestNodes, byzantinePower, honestPower, threshold);
    }

    function updateChart(byzNodes, honestNodes, byzPower, honestPower, threshold) {
        const canvas = document.getElementById('byzantine-chart');
        const ctx = canvas.getContext('2d');

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw power bars
        const maxPower = Math.max(byzPower, honestPower, threshold) * 1.2;
        const barWidth = 150;
        const barSpacing = 200;

        // Byzantine power bar
        ctx.fillStyle = '#f44336';
        const byzHeight = (byzPower / maxPower) * 250;
        ctx.fillRect(50, 250 - byzHeight, barWidth, byzHeight);
        ctx.fillStyle = '#fff';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Byzantine Power', 125, 280);
        ctx.fillText(byzPower.toFixed(2), 125, 295);

        // Honest power bar
        ctx.fillStyle = '#00e676';
        const honestHeight = (honestPower / maxPower) * 250;
        ctx.fillRect(250, 250 - honestHeight, barWidth, honestHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText('Honest Power', 325, 280);
        ctx.fillText(honestPower.toFixed(2), 325, 295);

        // Threshold line
        ctx.strokeStyle = '#ff9800';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        const thresholdY = 250 - (threshold / maxPower) * 250;
        ctx.beginPath();
        ctx.moveTo(0, thresholdY);
        ctx.lineTo(600, thresholdY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#ff9800';
        ctx.fillText(`Threshold: ${threshold.toFixed(2)}`, 500, thresholdY - 10);
    }

    // Add event listeners
    Object.values(inputs).forEach(input => {
        input.addEventListener('input', updateCalculation);
    });

    // Initial calculation
    updateCalculation();
}

// ============================================================================
// Trust Score Simulator
// ============================================================================

function createTrustScoreSimulator() {
    const container = document.getElementById('trust-score-simulator');
    if (!container) return;

    container.innerHTML = `
        <div class="interactive-widget">
            <h3>📊 Trust Score Evolution Simulator</h3>
            <p>See how trust scores evolve over training rounds</p>

            <div class="widget-controls">
                <div class="control-group">
                    <label for="num-nodes-trust">Number of Nodes:</label>
                    <input type="range" id="num-nodes-trust" min="3" max="10" value="5" step="1">
                    <span id="num-nodes-trust-value">5</span>
                </div>

                <div class="control-group">
                    <label for="attack-node">Attack Starting Round:</label>
                    <input type="range" id="attack-node" min="0" max="20" value="0" step="1">
                    <span id="attack-node-value">No attack</span>
                </div>

                <button id="run-simulation" class="widget-button">▶ Run Simulation</button>
                <button id="reset-simulation" class="widget-button">↻ Reset</button>
            </div>

            <div class="widget-visualization">
                <canvas id="trust-score-chart" width="700" height="400"></canvas>
            </div>

            <div class="widget-results">
                <div id="simulation-log"></div>
            </div>
        </div>
    `;

    let simulationData = [];
    let isRunning = false;
    let chart = null;

    const numNodesInput = document.getElementById('num-nodes-trust');
    const attackNodeInput = document.getElementById('attack-node');
    const runButton = document.getElementById('run-simulation');
    const resetButton = document.getElementById('reset-simulation');

    numNodesInput.addEventListener('input', () => {
        document.getElementById('num-nodes-trust-value').textContent = numNodesInput.value;
    });

    attackNodeInput.addEventListener('input', () => {
        const value = parseInt(attackNodeInput.value);
        document.getElementById('attack-node-value').textContent =
            value === 0 ? 'No attack' : `Round ${value}`;
    });

    runButton.addEventListener('click', runSimulation);
    resetButton.addEventListener('click', resetSimulation);

    function runSimulation() {
        if (isRunning) return;

        isRunning = true;
        runButton.disabled = true;

        const numNodes = parseInt(numNodesInput.value);
        const attackRound = parseInt(attackNodeInput.value);
        const maxRounds = 20;

        // Initialize nodes with random trust scores
        simulationData = Array(numNodes).fill(0).map((_, i) => ({
            id: i,
            trustScores: [0.5 + Math.random() * 0.1],
            isAttacker: i === numNodes - 1 && attackRound > 0,
            attackStartRound: attackRound,
        }));

        const log = document.getElementById('simulation-log');
        log.innerHTML = '<h4>Simulation Log:</h4>';

        let currentRound = 1;
        const interval = setInterval(() => {
            if (currentRound >= maxRounds) {
                clearInterval(interval);
                isRunning = false;
                runButton.disabled = false;
                log.innerHTML += '<p><strong>✅ Simulation complete!</strong></p>';
                return;
            }

            // Simulate round
            simulationData.forEach(node => {
                const lastScore = node.trustScores[node.trustScores.length - 1];
                let newScore;

                if (node.isAttacker && currentRound >= node.attackStartRound) {
                    // Attacker loses trust
                    newScore = Math.max(0.05, lastScore - 0.15 - Math.random() * 0.1);
                    if (currentRound === node.attackStartRound) {
                        log.innerHTML += `<p>⚠️ Round ${currentRound}: Node ${node.id} starts attacking!</p>`;
                    }
                } else {
                    // Honest node gains trust
                    newScore = Math.min(0.95, lastScore + 0.05 + Math.random() * 0.03);
                }

                node.trustScores.push(newScore);
            });

            // Update chart
            updateTrustChart(currentRound);

            currentRound++;
        }, 300);
    }

    function resetSimulation() {
        simulationData = [];
        document.getElementById('simulation-log').innerHTML = '';
        if (chart) {
            chart.destroy();
            chart = null;
        }
    }

    function updateTrustChart(round) {
        const canvas = document.getElementById('trust-score-chart');
        const ctx = canvas.getContext('2d');

        const colors = ['#2196f3', '#00e676', '#ff9800', '#e91e63', '#9c27b0', '#00bcd4', '#4caf50', '#ffeb3b', '#795548', '#607d8b'];

        // Create Chart.js datasets
        const datasets = simulationData.map((node, idx) => ({
            label: `Node ${idx}${node.isAttacker ? ' ⚠️ Attacker' : ''}`,
            data: node.trustScores,
            borderColor: colors[idx % colors.length],
            backgroundColor: colors[idx % colors.length] + '20',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 2,
            pointHoverRadius: 5,
        }));

        // Destroy previous chart if exists
        if (chart) {
            chart.data.datasets = datasets;
            chart.data.labels = Array.from({length: round + 1}, (_, i) => i);
            chart.update('none'); // No animation for smooth updates
        } else {
            // Create new Chart.js chart
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: round + 1}, (_, i) => i),
                    datasets: datasets
                },
                options: {
                    responsive: false,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'right',
                            labels: {
                                color: '#fff',
                                font: {
                                    size: 12
                                }
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(3);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Training Round',
                                color: '#fff'
                            },
                            ticks: {
                                color: '#fff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Trust Score',
                                color: '#fff'
                            },
                            min: 0,
                            max: 1,
                            ticks: {
                                color: '#fff',
                                callback: function(value) {
                                    return value.toFixed(1);
                                }
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
        }
    }
}

// ============================================================================
// Attack Type Comparison
// ============================================================================

function createAttackComparison() {
    const container = document.getElementById('attack-comparison');
    if (!container) return;

    const attacks = [
        { name: 'Sign Flip', detection: 98, impact: 'High', color: '#f44336' },
        { name: 'Gaussian Noise', detection: 95, impact: 'Medium', color: '#ff9800' },
        { name: 'Model Poisoning', detection: 92, impact: 'High', color: '#e91e63' },
        { name: 'Sleeper Agent', detection: 87, impact: 'Very High', color: '#9c27b0' },
        { name: 'Cartel', detection: 90, impact: 'Critical', color: '#f44336' },
        { name: 'Data Poisoning', detection: 93, impact: 'Medium', color: '#ff9800' },
        { name: 'Backdoor', detection: 89, impact: 'Critical', color: '#d32f2f' },
    ];

    let html = `
        <div class="interactive-widget">
            <h3>⚔️ Attack Type Comparison</h3>
            <p>Click on attack types to see details</p>
            <div class="attack-grid">
    `;

    attacks.forEach(attack => {
        html += `
            <div class="attack-card" data-attack="${attack.name}">
                <div class="attack-bar" style="width: ${attack.detection}%; background-color: ${attack.color}"></div>
                <div class="attack-info">
                    <h4>${attack.name}</h4>
                    <p>Detection: ${attack.detection}%</p>
                    <p>Impact: ${attack.impact}</p>
                </div>
            </div>
        `;
    });

    html += `
            </div>
            <div id="attack-details" class="attack-details"></div>
        </div>
    `;

    container.innerHTML = html;

    // Add click handlers
    document.querySelectorAll('.attack-card').forEach(card => {
        card.addEventListener('click', () => {
            const attackName = card.dataset.attack;
            showAttackDetails(attackName);
        });
    });
}

function showAttackDetails(attackName) {
    const details = {
        'Sign Flip': 'Malicious node flips the sign of all gradients, attempting to move the model in the opposite direction.',
        'Gaussian Noise': 'Adds random Gaussian noise to gradients to degrade model quality.',
        'Model Poisoning': 'Targets specific classes or samples to create backdoors in the model.',
        'Sleeper Agent': 'Behaves honestly for several rounds, then suddenly attacks when trust is high.',
        'Cartel': 'Multiple nodes coordinate their attacks to amplify impact.',
        'Data Poisoning': 'Uses intentionally mislabeled or corrupted training data.',
        'Backdoor': 'Embeds hidden triggers that cause misclassification on specific inputs.',
    };

    const detailsDiv = document.getElementById('attack-details');
    detailsDiv.innerHTML = `
        <h4>${attackName} Attack</h4>
        <p>${details[attackName]}</p>
        <p><strong>MATL Defense:</strong> Reputation-weighted validation automatically detects and downweights this attack type.</p>
    `;
}

// ============================================================================
// Back to Top Button
// ============================================================================

function createBackToTop() {
    const button = document.createElement('button');
    button.id = 'back-to-top';
    button.innerHTML = '↑';
    button.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-size: 24px;
        cursor: pointer;
        opacity: 0;
        transform: translateY(100px);
        transition: opacity 0.3s, transform 0.3s;
        box-shadow: 0 4px 12px rgba(103, 126, 234, 0.4);
    `;

    document.body.appendChild(button);

    // Show/hide based on scroll position
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            button.style.opacity = '1';
            button.style.transform = 'translateY(0)';
        } else {
            button.style.opacity = '0';
            button.style.transform = 'translateY(100px)';
        }
    });

    // Scroll to top on click
    button.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Hover effect
    button.addEventListener('mouseenter', () => {
        button.style.transform = 'translateY(-5px) scale(1.05)';
        button.style.boxShadow = '0 6px 16px rgba(103, 126, 234, 0.6)';
    });

    button.addEventListener('mouseleave', () => {
        if (window.scrollY > 300) {
            button.style.transform = 'translateY(0)';
            button.style.boxShadow = '0 4px 12px rgba(103, 126, 234, 0.4)';
        }
    });
}

// ============================================================================
// Performance: Lazy Widget Initialization
// ============================================================================

function lazyInitializeWidgets() {
    // Use Intersection Observer for lazy initialization
    const widgetContainers = [
        { id: 'byzantine-calculator', init: createByzantineCalculator },
        { id: 'trust-score-simulator', init: createTrustScoreSimulator },
        { id: 'attack-comparison', init: createAttackComparison },
    ];

    // Check if IntersectionObserver is supported
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const container = widgetContainers.find(w => w.id === entry.target.id);
                    if (container && !entry.target.dataset.initialized) {
                        container.init();
                        entry.target.dataset.initialized = 'true';
                        observer.unobserve(entry.target);
                    }
                }
            });
        }, {
            rootMargin: '100px', // Load 100px before entering viewport
            threshold: 0.1
        });

        // Observe all widget containers
        widgetContainers.forEach(widget => {
            const element = document.getElementById(widget.id);
            if (element) {
                observer.observe(element);
            }
        });
    } else {
        // Fallback for browsers without IntersectionObserver
        widgetContainers.forEach(widget => {
            if (document.getElementById(widget.id)) {
                widget.init();
            }
        });
    }
}

// ============================================================================
// Initialize all widgets
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    lazyInitializeWidgets();
    createBackToTop();
});
