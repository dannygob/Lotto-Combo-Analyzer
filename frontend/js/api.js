/**
 * LottoApiClient - REST API client for lottery datasets, models, metrics and predictions.
 * All methods return Promises and wrap the internal request() handler.
 */
class LottoApiClient {

    /**
     * Import a CSV dataset for a specific lottery.
     * @param {string} lotteryId - Lottery identifier.
     * @param {FormData} formData - CSV file payload.
     */
    async importCsv(lotteryId, formData) {
        return this.request(`/datasets/${lotteryId}/import`, {
            method: 'POST',
            body: formData
        });
    }

    /**
     * Save a CSV schema profile for a lottery.
     * @param {string} lotteryId - Lottery identifier.
     * @param {string} rulesetVersion - Version of the ruleset.
     * @param {Object} profile - Schema profile definition.
     */
    async saveSchemaProfile(lotteryId, rulesetVersion, profile) {
        return this.request(`/lotteries/${lotteryId}/csv-schema-profiles/${rulesetVersion}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile)
        });
    }

    // ============================================================================
    // SCHEDULE & PREDICTIONS
    // ============================================================================

    /**
     * Get the next scheduled draw for a lottery.
     */
    async getNextDraw(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/next-draw`);
    }

    /**
     * Get prediction results for a lottery.
     * @param {string} lotteryId
     * @param {number} candidateLimit - Max number of candidate combinations.
     */
    async getPrediction(lotteryId, candidateLimit = 5) {
        return this.request(`/lotteries/${lotteryId}/prediction?candidate_limit=${candidateLimit}`);
    }

    // ============================================================================
    // MODEL GOVERNANCE
    // ============================================================================

    /**
     * Retrieve all available models for a lottery.
     */
    async getModels(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/models`);
    }

    /**
     * Promote a model version to active status.
     */
    async promoteModel(modelVersion) {
        return this.request(`/models/${modelVersion}/promote`, {
            method: 'POST'
        });
    }

    // ============================================================================
    // FREQUENCY, CLUSTERS, RULES AND METRICS
    // ============================================================================

    /**
     * Get global frequency metrics for a lottery.
     */
    async getFrequency(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/frequency`);
    }

    /**
     * Get clustering results for a lottery.
     * @param {number} nClusters - Number of clusters to compute.
     */
    async getClusters(lotteryId, nClusters = 5) {
        return this.request(`/lotteries/${lotteryId}/clusters?n_clusters=${nClusters}`);
    }

    /**
     * Get association rules (Apriori / FP-Growth).
     */
    async getAssociationRules(lotteryId, topN = 10) {
        return this.request(`/lotteries/${lotteryId}/association-rules?top_n=${topN}`);
    }

    /**
     * Get model performance metrics.
     */
    async getModelMetrics(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/model-metrics`);
    }

    /**
     * Get correlation matrix for lottery numbers.
     */
    async getCorrelationMatrix(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/correlation-matrix`);
    }

    // ============================================================================
    // POOL METRIC FUNCTIONS
    // ============================================================================

    /**
     * Get metrics for a specific pool.
     */
    async getPoolMetrics(lotteryId, poolId) {
        return this.request(`/lotteries/${lotteryId}/pool-metrics/${poolId}`);
    }

    /**
     * Get metrics for all pools.
     */
    async getAllPoolMetrics(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/all-pool-metrics`);
    }

    /**
     * Get frequency metrics for a specific pool.
     */
    async getPoolFrequency(lotteryId, poolId) {
        return this.request(`/lotteries/${lotteryId}/pool-frequency/${poolId}`);
    }

    /**
     * Get frequency metrics for all pools.
     */
    async getAllPoolFrequency(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/all-pool-frequency`);
    }

    // ============================================================================
    // MODEL IMPROVEMENT AND SUM DISTRIBUTION
    // ============================================================================

    /**
     * Trigger model improvement pipeline.
     */
    async improveModel(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/improve-model`, {
            method: 'POST',
        });
    }

    /**
     * Retrieve historical model improvements.
     */
    async getModelImprovements(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/model-improvements`);
    }

    /**
     * Get sum distribution for a specific pool.
     */
    async getSumDistribution(lotteryId, poolId) {
        return this.request(`/lotteries/${lotteryId}/sum-distribution/${poolId}`);
    }

    /**
     * Validate sum distribution predictions.
     */
    async getSumValidation(lotteryId, poolId) {
        return this.request(`/lotteries/${lotteryId}/sum-validation/${poolId}`);
    }
}

// Expose API client globally
window.apiClient = new LottoApiClient();
