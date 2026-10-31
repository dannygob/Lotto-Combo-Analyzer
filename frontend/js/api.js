
    async importCsv(lotteryId, formData) {
        return this.request(`/datasets/${lotteryId}/import`, {
            method: 'POST',
            body: formData
        });
    }

    async saveSchemaProfile(lotteryId, rulesetVersion, profile) {
        return this.request(`/lotteries/${lotteryId}/csv-schema-profiles/${rulesetVersion}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile)
        });
    }

    // Schedule & Predictions
    async getNextDraw(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/next-draw`);
    }

    async getPrediction(lotteryId, candidateLimit = 5) {
        return this.request(`/lotteries/${lotteryId}/prediction?candidate_limit=${candidateLimit}`);
    }

    // Model Governance
    async getModels(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/models`);
    }

    async promoteModel(modelVersion) {
        return this.request(`/models/${modelVersion}/promote`, {
            method: 'POST'
        });
    }

    // --- NUEVOS ENDPOINTS PARA FRECUENCIA, CLUSTERS, REGLAS Y MÉTRICAS ---
    async getFrequency(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/frequency`);
    }

    async getClusters(lotteryId, nClusters = 5) {
        return this.request(`/lotteries/${lotteryId}/clusters?n_clusters=${nClusters}`);
    }

    async getAssociationRules(lotteryId, topN = 10) {
        return this.request(`/lotteries/${lotteryId}/association-rules?top_n=${topN}`);
    }

    async getModelMetrics(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/model-metrics`);
    }

    async getCorrelationMatrix(lotteryId) {
        return this.request(`/lotteries/${lotteryId}/correlation-matrix`);
    }
}

window.apiClient = new LottoApiClient();