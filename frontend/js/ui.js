/**
 * Lotto-Combo-Analyzer - UI Renderer Component
 * Handles dynamic DOM generation for balls, predictions, tables, and metrics.
 */
let UI = {
    /**
     * Render a single lottery ball element.
     * @param {number|string} number - The ball value to display.
     * @param {string} type - Visual type: 'primary' or 'bonus'.
     * @returns {string} HTML string for the ball.
     */
    renderBall(number, type = 'primary') {
        const ballClass = `lotto-ball lotto-ball-${type}`;
        return `<span class="${ballClass}">${number}</span>`;
    },

    /**
     * Render candidate combinations with probability and score badges.
     * @param {Array} candidates - List of candidate objects.
     * @param {string} poolKind - Pool type: 'primary' or 'bonus'.
     * @returns {string} HTML string containing all candidate rows.
     */
    renderCandidates(candidates, poolKind = 'primary') {
        if (!candidates || candidates.length === 0) {
            return '<p class="text-slate-500 text-xs">No candidates available.</p>';
        }

        return candidates.map(c => {
            const ballsHtml = c.numbers.map(num => this.renderBall(num, poolKind)).join(' ');
            const probPercent = (c.model_implied_combination_probability * 100).toFixed(4);

            return `
                <div class="bg-dark-900 border border-dark-700 p-3 rounded-lg flex flex-col md:flex-row justify-between items-center gap-3 mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-xs font-bold text-slate-400 bg-dark-800 px-2 py-1 rounded">#${c.rank}</span>
                        <div class="flex flex-wrap gap-1">${ballsHtml}</div>
                    </div>
                    <div class="text-right">
                        <span class="text-xs font-mono text-indigo-400 font-semibold">P(implied): ${probPercent}%</span>
                        <span class="text-[10px] text-slate-500 block">Score: ${c.score.toFixed(3)}</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Render the full prediction results for all pools.
     * @param {string} containerId - DOM element ID where results will be injected.
     * @param {Object} predictionData - Prediction payload returned by the model.
     */
    renderPredictionResults(containerId, predictionData) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Validate prediction payload
        if (!predictionData || predictionData.status !== 'research_prediction') {
            container.innerHTML = `<div class="p-4 text-red-400 text-xs">Prediction data error.</div>`;
            return;
        }

        const { event, model_version, ruleset_version, pools, scientific_note } = predictionData;

        let poolsHtml = '';
        for (const [poolId, poolData] of Object.entries(pools)) {
            poolsHtml += `
                <div class="bg-dark-800 border border-dark-700 rounded-xl p-5 mb-4 shadow-lg">
                    <div class="flex justify-between items-center mb-3">
                        <h3 class="text-md font-bold text-indigo-400 uppercase tracking-wide">Pool: ${poolId}</h3>
                        <span class="text-xs bg-indigo-900/50 text-indigo-300 border border-indigo-700 px-2 py-0.5 rounded">
                            ${poolData.candidates.length} Candidates
                        </span>
                    </div>
                    <div class="space-y-2">
                        ${this.renderCandidates(poolData.candidates, poolId.includes('bonus') ? 'bonus' : 'primary')}
                    </div>
                </div>
            `;
        }

        // Final UI injection
        container.innerHTML = `
            <div class="space-y-4">
                <!-- Metadata Header -->
                <div class="bg-dark-800/80 border border-dark-700 p-4 rounded-xl flex flex-wrap justify-between items-center text-xs text-slate-300 gap-2">
                    <div><strong>Event ID:</strong> <span class="text-indigo-400">${event.event_id}</span></div>
                    <div><strong>Date:</strong> ${event.event_date}</div>
                    <div><strong>Active Model:</strong> <span class="text-emerald-400">${model_version}</span> (Ruleset: ${ruleset_version})</div>
                </div>

                <!-- Pools Section -->
                ${poolsHtml}

                <!-- Scientific Disclaimer -->
                <div class="p-3 bg-dark-900 border border-dark-700/60 rounded-lg text-[11px] text-slate-400 italic">
                    ⚠️ <strong>Scientific Note:</strong> ${scientific_note}
                </div>
            </div>
        `;
    }
};

// Expose UI globally
window.UI = UI;
