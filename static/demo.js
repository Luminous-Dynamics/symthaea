// Symthaea Live Consciousness Demo — WebSocket Client
// No build tooling, vanilla JS + Chart.js

const WINDOW_SIZE = 200;
const WS_URL = `ws://${window.location.host}/v1/ws/live`;

// --- State ---
let ws = null;
let paused = false;
const history = {
    cycles: [],
    consciousness: [],
    psi: [],
    predictionError: [],
    valence: [],
    arousal: [],
    coherence: [],
    moral: [],
};

// --- Chart Setup ---
const chartOptions = (yMin, yMax) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { legend: { display: true, labels: { color: '#9ca3af', font: { size: 10 } } } },
    scales: {
        x: { display: false },
        y: {
            min: yMin,
            max: yMax,
            ticks: { color: '#6b7280', font: { size: 9 } },
            grid: { color: '#1f2937' },
        },
    },
});

const makeDataset = (label, color, data) => ({
    label,
    data,
    borderColor: color,
    backgroundColor: color + '20',
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
});

const chartPsi = new Chart(document.getElementById('chartPsi'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Consciousness', '#06b6d4', history.consciousness),
            makeDataset('Psi', '#a855f7', history.psi),
        ],
    },
    options: chartOptions(0, 1),
});

const chartError = new Chart(document.getElementById('chartError'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [makeDataset('Prediction Error', '#f59e0b', history.predictionError)],
    },
    options: chartOptions(0, undefined),
});

const chartAffect = new Chart(document.getElementById('chartAffect'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Valence', '#22c55e', history.valence),
            makeDataset('Arousal', '#f43f5e', history.arousal),
        ],
    },
    options: chartOptions(-1, 1),
});

const chartCoherence = new Chart(document.getElementById('chartCoherence'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            makeDataset('Coherence', '#06b6d4', history.coherence),
            makeDataset('Moral Score', '#a855f7', history.moral),
        ],
    },
    options: chartOptions(0, 1),
});

// --- WebSocket ---
function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        document.getElementById('statusDot').classList.remove('disconnected');
        document.getElementById('statusText').textContent = 'Connected';
    };

    ws.onclose = () => {
        document.getElementById('statusDot').classList.add('disconnected');
        document.getElementById('statusText').textContent = 'Disconnected — reconnecting...';
        setTimeout(connect, 2000);
    };

    ws.onerror = () => {
        ws.close();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'connected') return; // Skip hello

        // Push to history (rolling window)
        const push = (arr, val) => {
            arr.push(val);
            if (arr.length > WINDOW_SIZE) arr.shift();
        };

        push(history.cycles, data.cycle);
        push(history.consciousness, data.consciousness_level);
        push(history.psi, data.narrative_self_psi);
        push(history.predictionError, data.prediction_error);
        push(history.valence, data.valence);
        push(history.arousal, data.arousal);
        push(history.coherence, data.coherence);
        push(history.moral, data.moral_score);

        // Update charts
        const labels = history.cycles.map(String);
        chartPsi.data.labels = labels;
        chartError.data.labels = labels;
        chartAffect.data.labels = labels;
        chartCoherence.data.labels = labels;

        chartPsi.update('none');
        chartError.update('none');
        chartAffect.update('none');
        chartCoherence.update('none');

        // Update metrics
        document.getElementById('metricCycle').textContent = data.cycle;
        document.getElementById('metricPsi').textContent = data.consciousness_level.toFixed(3);
        document.getElementById('metricError').textContent = data.prediction_error.toFixed(3);
        document.getElementById('metricMoral').textContent = data.moral_score.toFixed(3);
        document.getElementById('metricTime').textContent = data.cycle_time_us.toLocaleString();
        document.getElementById('metricReasoning').textContent = data.reasoning_confidence.toFixed(2);

        // Update state flags
        const setFlag = (id, active) => {
            const el = document.getElementById(id);
            el.classList.toggle('active', active);
        };
        setFlag('flagSurprise', data.surprise_triggered);
        setFlag('flagGWT', data.gwt_broadcast);
        setFlag('flagDream', data.dream_insights > 0);
        setFlag('flagReasoning', data.reasoning_confidence > 0);

        // Update cycle info
        document.getElementById('cycleInfo').textContent =
            `Cycle ${data.cycle} | ${(data.cycle_time_us / 1000).toFixed(1)}ms | Input: "${data.input_text}"`;
    };
}

// --- Controls ---
document.getElementById('btnSend').addEventListener('click', () => {
    const input = document.getElementById('textInput');
    if (ws && ws.readyState === WebSocket.OPEN && input.value.trim()) {
        ws.send(JSON.stringify({ text: input.value.trim() }));
        input.value = '';
    }
});

document.getElementById('textInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') document.getElementById('btnSend').click();
});

document.getElementById('btnPause').addEventListener('click', () => {
    paused = !paused;
    const btn = document.getElementById('btnPause');
    btn.textContent = paused ? 'Resume' : 'Pause';
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: paused ? 'pause' : 'resume' }));
    }
});

document.getElementById('btnReset').addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: 'reset' }));
    }
    // Clear history
    Object.values(history).forEach(arr => arr.length = 0);
});

// --- Start ---
connect();
