const API_BASE = 'http://localhost:8000';
let registeredLotteries = [];
let editingLotteryId = null;
let selectedRulesetVersion = null;
let selectedRuleset = null;
let savedCsvProfile = null;
let inspectedCsv = null;
let availableFormats = [];

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    })[character]);
}

// Control de Pestañas (Tabs)
function switchTab(tabId) {
    document.querySelectorAll('main > section').forEach(sec => sec.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-indigo-600', 'text-white', 'shadow-lg');
        btn.classList.add('text-slate-400', 'hover:text-white', 'hover:bg-dark-800');
    });

    document.getElementById(`tab-${tabId}`).classList.remove('hidden');
    const activeBtn = document.getElementById(`btn-${tabId}`);
    activeBtn.classList.remove('text-slate-400', 'hover:text-white', 'hover:bg-dark-800');
    activeBtn.classList.add('bg-indigo-600', 'text-white', 'shadow-lg');
}

// Verificación de Estado de Salud de la API
async function checkHealth() {
    const dot = document.getElementById('health-dot');
    const text = document.getElementById('health-text');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === 'ok') {
            dot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500';
            text.textContent = 'API Conectada';
            loadLotteries();
        } else {
            dot.className = 'w-2.5 h-2.5 rounded-full bg-red-500';
            text.textContent = 'Error en API';
        }
    } catch (err) {
        dot.className = 'w-2.5 h-2.5 rounded-full bg-red-500';
        text.textContent = 'Sin conexión con FastAPI';
    }
}

// Cargar Loterías Registradas
async function loadLotteries() {
    const container = document.getElementById('lotteries-container');
    try {
        const res = await fetch(`${API_BASE}/lotteries`);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const lotteries = await res.json();
        registeredLotteries = lotteries;
        await loadImportLotteries();
        if (!lotteries || lotteries.length === 0) {
            container.innerHTML = '<p class="text-slate-500 text-sm">No hay loterías registradas.</p>';
            return;
        }
        container.innerHTML = lotteries.map(l => `
            <div class="bg-dark-900 border border-dark-700 p-3 rounded-lg flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 text-sm">
                <div class="min-w-0">
                    <span class="font-bold text-indigo-400">${escapeHtml(l.name)}</span>
                    <p class="text-xs text-slate-500">ID interno: ${escapeHtml(l.lottery_id)}</p>
                </div>
                <div class="flex gap-2 shrink-0">
                    <button type="button" data-action="edit" data-lottery-id="${escapeHtml(l.lottery_id)}" class="text-xs bg-dark-700 hover:bg-dark-600 px-3 py-1.5 rounded transition">Editar</button>
                    <button type="button" data-action="delete" data-lottery-id="${escapeHtml(l.lottery_id)}" class="text-xs bg-red-900/70 hover:bg-red-800 text-red-100 px-3 py-1.5 rounded transition">Eliminar</button>
                </div>
            </div>
        `).join('');
    } catch (err) {
        container.innerHTML = `<p class="text-red-400 text-sm">Error cargando loterías: ${escapeHtml(err.message)}</p>`;
    }
}

async function loadImportLotteries() {
    const select = document.getElementById('import-lottery-select');
    const previousValue = select.value;
    select.replaceChildren(new Option('Selecciona una lotería', ''));
    registeredLotteries.forEach(lottery => select.add(new Option(lottery.name, lottery.lottery_id)));
    if (registeredLotteries.some(item => item.lottery_id === previousValue)) {
        select.value = previousValue;
    } else if (registeredLotteries.length === 1) {
        select.value = registeredLotteries[0].lottery_id;
    }

    if (select.value) {
        await loadLotteryFormats(select.value);
    } else {
        resetFormatSelection();
    }
}

function resetFormatSelection() {
    selectedRulesetVersion = null;
    selectedRuleset = null;
    savedCsvProfile = null;
    availableFormats = [];
    document.getElementById('format-existing').classList.add('hidden');
    document.getElementById('format-setup').classList.add('hidden');
    document.getElementById('format-current').classList.add('hidden');
    document.getElementById('import-actions').classList.add('hidden');
    document.getElementById('csv-column-mapping').classList.add('hidden');
}

async function loadLotteryFormats(lotteryId) {
    resetFormatSelection();
    const resultBox = document.getElementById('import-result');
    try {
        const res = await fetch(`${API_BASE}/lotteries/${encodeURIComponent(lotteryId)}/rulesets`);
        if (res.status === 404) {
            availableFormats = [];
            showFormatSetupIfReady();
            return;
        }
        const formats = await res.json();
        if (!res.ok) {
            throw new Error(formats.detail || `HTTP ${res.status}`);
        }
        availableFormats = formats;
        const select = document.getElementById('format-existing-select');
        select.replaceChildren(...formats.map(format => new Option(
            `Formato ${format.version} (desde ${format.effective_from})`,
            format.version
        )));
        document.getElementById('format-existing').classList.remove('hidden');
        document.getElementById('format-setup').classList.add('hidden');
        await selectSavedFormat(select.value);
    } catch (err) {
        resultBox.classList.remove('hidden');
        resultBox.textContent = `No se pudieron cargar los formatos de la lotería: ${err.message}`;
    }
}

async function selectSavedFormat(version) {
    const lotteryId = document.getElementById('import-lottery-select').value;
    if (!lotteryId || !version) {
        return;
    }
    const resultBox = document.getElementById('import-result');
    try {
        const res = await fetch(
            `${API_BASE}/lotteries/${encodeURIComponent(lotteryId)}/rulesets/${encodeURIComponent(version)}`
        );
        const ruleset = await res.json();
        if (!res.ok) {
            throw new Error(ruleset.detail || `HTTP ${res.status}`);
        }
        selectedRulesetVersion = ruleset.version;
        selectedRuleset = ruleset;
        savedCsvProfile = null;
        document.getElementById('chronology-confirmed').checked = false;
        document.getElementById('format-current').textContent = ruleset.pools.map(pool =>
            `Formato ${ruleset.version}: ${pool.numbers_drawn} números por sorteo, del ${pool.minimum} al ${pool.maximum}.`
        ).join(' ');
        document.getElementById('format-current').classList.remove('hidden');
        document.getElementById('format-setup').classList.add('hidden');
        if (inspectedCsv) {
            try {
                const profileResponse = await fetch(
                    `${API_BASE}/lotteries/${encodeURIComponent(lotteryId)}/csv-schema-profiles/${encodeURIComponent(version)}`
                );
                if (profileResponse.ok) {
                    savedCsvProfile = await profileResponse.json();
                    document.getElementById('chronology-confirmed').checked =
                        savedCsvProfile.chronological_order_confirmed;
                } else if (profileResponse.status !== 404) {
                    const profileError = await profileResponse.json();
                    throw new Error(profileError.detail || `HTTP ${profileResponse.status}`);
                }
            } catch (err) {
                throw new Error(`No se pudo cargar la asignación guardada: ${err.message}`);
            }
            renderCsvColumnMapping();
        }
        document.getElementById('import-actions').classList.toggle(
            'hidden',
            document.getElementById('import-file').files.length === 0
        );
    } catch (err) {
        resultBox.classList.remove('hidden');
        resultBox.textContent = `No se pudo cargar el formato guardado: ${err.message}`;
    }
}

function renderCsvColumnMapping() {
    const mapping = document.getElementById('csv-column-mapping');
    const roles = document.getElementById('csv-column-roles');
    if (!inspectedCsv || !selectedRuleset) {
        mapping.classList.add('hidden');
        return;
    }

    const pools = selectedRuleset.pools;
    const suggestedPools = {};
    let numberIndex = 0;
    for (const pool of pools) {
        suggestedPools[pool.pool_id] = (inspectedCsv.number_columns || [])
            .slice(numberIndex, numberIndex + pool.numbers_drawn);
        numberIndex += pool.numbers_drawn;
    }
    const existingPoolColumns = savedCsvProfile?.pool_columns || {};
    const existingRoles = new Map();
    for (const [poolId, columns] of Object.entries(existingPoolColumns)) {
        columns.forEach(column => existingRoles.set(column, `pool:${poolId}`));
    }
    for (const [column, value] of [
        [savedCsvProfile?.date_column, 'date'],
        [savedCsvProfile?.event_id_column, 'event-id'],
        [savedCsvProfile?.session_column, 'session'],
        [savedCsvProfile?.draw_time_column, 'draw-time']
    ]) {
        if (column) existingRoles.set(column, value);
    }
    (savedCsvProfile?.allowed_attribute_columns || []).forEach(column =>
        existingRoles.set(column, 'attribute')
    );
    (savedCsvProfile?.ignored_columns || []).forEach(column =>
        existingRoles.set(column, 'ignore')
    );

    const columns = inspectedCsv.columns || [];
    roles.innerHTML = columns.map(column => {
        let suggestedRole = 'attribute';
        if ((inspectedCsv.date_columns || []).includes(column)) {
            suggestedRole = 'date';
        } else if ((inspectedCsv.event_id_columns || []).includes(column)) {
            suggestedRole = 'event-id';
        } else {
            const suggestedPool = pools.find(pool => suggestedPools[pool.pool_id]?.includes(column));
            if (suggestedPool) {
                suggestedRole = `pool:${suggestedPool.pool_id}`;
            }
        }
        const selectedRole = existingRoles.get(column) || suggestedRole;
        const options = [
            ...pools.map(pool => ({
                value: `pool:${pool.pool_id}`,
                label: `Número sorteado (${pool.kind === 'primary' ? 'principal' : pool.kind})`
            })),
            { value: 'date', label: 'Fecha del sorteo' },
            { value: 'event-id', label: 'Número/ID del sorteo' },
            { value: 'session', label: 'Turno o sesión' },
            { value: 'draw-time', label: 'Hora del sorteo' },
            { value: 'attribute', label: 'Guardar como dato adicional' },
            { value: 'ignore', label: 'Ignorar' }
        ];
        return `
            <label class="grid grid-cols-1 sm:grid-cols-2 gap-2 items-center text-xs">
                <span class="text-slate-300 break-all">${escapeHtml(column)}</span>
                <select data-csv-column="${escapeHtml(column)}" class="bg-dark-800 border border-dark-700 rounded px-2 py-2 text-xs">
                    ${options.map(option => `
                        <option value="${escapeHtml(option.value)}" ${selectedRole === option.value ? 'selected' : ''}>
                            ${escapeHtml(option.label)}
                        </option>
                    `).join('')}
                </select>
            </label>
        `;
    }).join('');
    mapping.classList.remove('hidden');
    updateChronologyConfirmationVisibility();
}

function updateChronologyConfirmationVisibility() {
    const dateSelected = [...document.querySelectorAll('#csv-column-roles select')]
        .some(select => select.value === 'date');
    document.getElementById('chronology-confirmation-container').classList.toggle('hidden', dateSelected);
}

function buildCsvSchemaProfile() {
    if (!inspectedCsv || !selectedRuleset || !selectedRulesetVersion) {
        throw new Error('Selecciona un CSV y el formato de la lotería antes de asignar columnas.');
    }
    const poolColumns = Object.fromEntries(
        selectedRuleset.pools.map(pool => [pool.pool_id, []])
    );
    const profile = {
        lottery_id: document.getElementById('import-lottery-select').value,
        ruleset_version: selectedRulesetVersion,
        pool_columns: poolColumns,
        date_column: null,
        event_id_column: null,
        session_column: null,
        draw_time_column: null,
        allowed_attribute_columns: [],
        ignored_columns: [],
        chronological_order_confirmed: false
    };
    const columns = document.querySelectorAll('#csv-column-roles select');
    for (const select of columns) {
        const column = select.dataset.csvColumn;
        const role = select.value;
        if (role.startsWith('pool:')) {
            const poolId = role.slice('pool:'.length);
            if (!profile.pool_columns[poolId]) {
                throw new Error(`La columna "${column}" está asignada a un grupo de números desconocido.`);
            }
            profile.pool_columns[poolId].push(column);
        } else if (role === 'date') {
            if (profile.date_column) throw new Error('Solo puedes asignar una columna como fecha.');
            profile.date_column = column;
        } else if (role === 'event-id') {
            if (profile.event_id_column) throw new Error('Solo puedes asignar una columna como número/ID del sorteo.');
            profile.event_id_column = column;
        } else if (role === 'session') {
            if (profile.session_column) throw new Error('Solo puedes asignar una columna como turno o sesión.');
            profile.session_column = column;
        } else if (role === 'draw-time') {
            if (profile.draw_time_column) throw new Error('Solo puedes asignar una columna como hora del sorteo.');
            profile.draw_time_column = column;
        } else if (role === 'attribute') {
            profile.allowed_attribute_columns.push(column);
        } else if (role === 'ignore') {
            profile.ignored_columns.push(column);
        }
    }
    for (const pool of selectedRuleset.pools) {
        const assigned = profile.pool_columns[pool.pool_id];
        if (assigned.length !== pool.numbers_drawn) {
            throw new Error(
                `El grupo "${pool.pool_id}" necesita ${pool.numbers_drawn} columnas de números; asignaste ${assigned.length}.`
            );
        }
    }
    if (!profile.date_column && !document.getElementById('chronology-confirmed').checked) {
        throw new Error('Elige una columna de fecha o confirma que el CSV está ordenado cronológicamente.');
    }
    profile.chronological_order_confirmed = !profile.date_column &&
        document.getElementById('chronology-confirmed').checked;
    return profile;
}

async function saveCsvSchemaProfile() {
    const profile = buildCsvSchemaProfile();
    const lotteryId = encodeURIComponent(profile.lottery_id);
    const version = encodeURIComponent(profile.ruleset_version);
    const res = await fetch(
        `${API_BASE}/lotteries/${lotteryId}/csv-schema-profiles/${version}`,
        {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(profile)
        }
    );
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
    }
    savedCsvProfile = profile;
    return profile;
}

function showFormatSetupIfReady() {
    const hasLottery = Boolean(document.getElementById('import-lottery-select').value);
    const hasFile = document.getElementById('import-file').files.length > 0;
    if (hasLottery && hasFile && !selectedRulesetVersion) {
        document.getElementById('format-setup').classList.remove('hidden');
        document.getElementById('import-actions').classList.add('hidden');
    }
}

async function inspectCsv() {
    const file = document.getElementById('import-file').files[0];
    const resultBox = document.getElementById('import-result');
    inspectedCsv = null;
    selectedRulesetVersion = null;
    document.getElementById('chronology-confirmed').checked = false;
    document.getElementById('format-current').classList.add('hidden');
    document.getElementById('import-actions').classList.add('hidden');
    if (!file) {
        document.getElementById('format-setup').classList.add('hidden');
        return;
    }

    resultBox.classList.remove('hidden');
    resultBox.textContent = 'Revisando las columnas y los valores del archivo...';
    const formData = new FormData();
    formData.append('file', file);
    try {
        const res = await fetch(`${API_BASE}/datasets/inspect`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || `HTTP ${res.status}`);
        }
        inspectedCsv = data;
        const suggestion = data.suggested_format;
        const columnsText = data.number_columns.length
            ? `Columnas que parecen contener números: ${data.number_columns.join(', ')}.`
            : 'No se reconocieron columnas con nombres como “numero 1” o “n1”.';
        const dateText = data.date_columns?.length
            ? ` Posible fecha: ${data.date_columns.join(', ')}.`
            : ' No se detectó una columna de fecha; puedes asignarla manualmente o confirmar el orden de las filas.';
        const drawNumberText = data.event_id_columns?.length
            ? ` Posible número de sorteo: ${data.event_id_columns.join(', ')}.`
            : '';
        document.getElementById('format-inspection').textContent = suggestion
            ? `${data.rows} filas detectadas. ${columnsText}${dateText}${drawNumberText} En este archivo aparecen valores del ${suggestion.minimum} al ${suggestion.maximum}, con ${suggestion.numbers_drawn} números por sorteo.`
            : `${data.rows} filas detectadas. ${columnsText}${dateText}${drawNumberText} Completa el formato manualmente.`;
        document.getElementById('format-minimum').value = suggestion?.minimum ?? '';
        document.getElementById('format-maximum').value = suggestion?.maximum ?? '';
        document.getElementById('format-count').value = suggestion?.numbers_drawn ?? '';
        const today = new Date();
        const localToday = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
        document.getElementById('format-effective-from').value = data.earliest_draw_date || localToday;
        document.getElementById('format-confirmed').checked = false;
        resultBox.textContent = `${data.warning} Revisa los datos sugeridos en “Formato del sorteo”.`;

        const lotteryId = document.getElementById('import-lottery-select').value;
        if (lotteryId) {
            await loadLotteryFormats(lotteryId);
            if (selectedRulesetVersion) {
                document.getElementById('import-actions').classList.remove('hidden');
            } else {
                showFormatSetupIfReady();
            }
        }
    } catch (err) {
        resultBox.textContent = `No se pudo revisar el CSV: ${err.message}`;
    }
}

async function saveDrawFormat() {
    const lotteryId = document.getElementById('import-lottery-select').value;
    const file = document.getElementById('import-file').files[0];
    const minimum = Number(document.getElementById('format-minimum').value);
    const maximum = Number(document.getElementById('format-maximum').value);
    const numbersDrawn = Number(document.getElementById('format-count').value);
    const effectiveFrom = document.getElementById('format-effective-from').value;
    const resultBox = document.getElementById('import-result');

    if (!lotteryId || !file || !inspectedCsv) {
        alert('Selecciona una lotería y un archivo CSV antes de configurar su formato.');
        return;
    }
    if (!document.getElementById('format-confirmed').checked) {
        alert('Confirma que los límites y la cantidad coinciden con las reglas oficiales del juego.');
        return;
    }
    if (!Number.isInteger(minimum) || !Number.isInteger(maximum) ||
        !Number.isInteger(numbersDrawn) || minimum < 0 || maximum < minimum ||
        numbersDrawn < 1 || numbersDrawn > maximum - minimum + 1 || !effectiveFrom) {
        alert('Revisa el rango de números, cuántos se sortean y desde qué fecha aplica el formato.');
        return;
    }

    const nextVersion = availableFormats.reduce((highest, format) => {
        const match = /^v(\d+)$/i.exec(format.version);
        return match ? Math.max(highest, Number(match[1])) : highest;
    }, 0) + 1;
    const payload = {
        lottery_id: lotteryId,
        version: `v${nextVersion}`,
        effective_from: effectiveFrom,
        pools: [{
            pool_id: 'main',
            kind: 'primary',
            minimum,
            maximum,
            numbers_drawn: numbersDrawn,
            allow_replacement: false
        }]
    };

    resultBox.classList.remove('hidden');
    resultBox.textContent = 'Guardando el formato del sorteo...';
    try {
        const res = await fetch(`${API_BASE}/rulesets`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || `HTTP ${res.status}`);
        }
        await loadLotteryFormats(lotteryId);
        document.getElementById('format-existing-select').value = data.version;
        await selectSavedFormat(data.version);
        resultBox.textContent = `Formato ${data.version} guardado. Ya puedes revisar e importar el historial.`;
    } catch (err) {
        resultBox.textContent = `No se pudo guardar el formato: ${err.message}`;
    }
}

function startEditLottery(lotteryId) {
    const lottery = registeredLotteries.find(item => item.lottery_id === lotteryId);
    if (!lottery) {
        alert('No se encontró la lotería. Actualiza la lista e inténtalo de nuevo.');
        return;
    }

    editingLotteryId = lotteryId;
    document.getElementById('lottery_name').value = lottery.name;
    document.getElementById('lottery-form-title').textContent = 'Editar Lotería';
    document.getElementById('lottery-submit').textContent = 'Guardar Cambios';
    document.getElementById('lottery-cancel-edit').classList.remove('hidden');
}

function cancelEditLottery() {
    editingLotteryId = null;
    document.getElementById('form-lottery').reset();
    document.getElementById('lottery-form-title').textContent = 'Registrar Nueva Lotería';
    document.getElementById('lottery-submit').textContent = 'Guardar Lotería';
    document.getElementById('lottery-cancel-edit').classList.add('hidden');
}

async function deleteLottery(lotteryId) {
    const lottery = registeredLotteries.find(item => item.lottery_id === lotteryId);
    const name = lottery ? lottery.name : lotteryId;
    const confirmed = confirm(
        `¿Eliminar "${name}" y TODOS sus datos relacionados?\n\n` +
        'Se borrarán reglas, sorteos, perfiles CSV, importaciones y modelos. Esta acción no se puede deshacer.'
    );
    if (!confirmed) {
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/lotteries/${encodeURIComponent(lotteryId)}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || `HTTP ${res.status}`);
        }
        if (editingLotteryId === lotteryId) {
            cancelEditLottery();
        }
        await loadLotteries();
        const cleanupWarning = data.cleanup_errors?.length
            ? `\nNo se pudieron eliminar algunos archivos: ${data.cleanup_errors.map(item => item.path).join(', ')}`
            : '';
        alert(`Lotería eliminada. Registros relacionados borrados: ${Object.values(data.deleted_counts || {}).reduce((sum, count) => sum + count, 0)}.${cleanupWarning}`);
    } catch (err) {
        alert(`No se pudo eliminar la lotería: ${err.message}`);
    }
}

document.getElementById('lotteries-container').addEventListener('click', event => {
    const button = event.target.closest('button[data-action]');
    if (!button) {
        return;
    }
    if (button.dataset.action === 'edit') {
        startEditLottery(button.dataset.lotteryId);
    } else if (button.dataset.action === 'delete') {
        deleteLottery(button.dataset.lotteryId);
    }
});

// Registrar Nueva Lotería
document.getElementById('form-lottery').addEventListener('submit', async (e) => {
    e.preventDefault();
    const isEditing = editingLotteryId !== null;
    const payload = {
        name: document.getElementById('lottery_name').value
    };
    try {
        const url = isEditing
            ? `${API_BASE}/lotteries/${encodeURIComponent(editingLotteryId)}`
            : `${API_BASE}/lotteries`;
        const res = await fetch(url, {
            method: isEditing ? 'PUT' : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            const data = await res.json();
            alert(isEditing
                ? '¡Lotería actualizada con éxito!'
                : `¡Lotería registrada con éxito! ID interno: ${data.lottery_id}`);
            cancelEditLottery();
            await loadLotteries();
        } else {
            const err = await res.json();
            alert(`Error: ${err.detail || 'No se pudo guardar la lotería'}`);
        }
    } catch (err) {
        alert('Error de conexión al guardar la lotería.');
    }
});

// Previsualizar Esquema CSV
async function previewCsv() {
    const lotteryId = document.getElementById('import-lottery-select').value;
    const rulesetVersion = selectedRulesetVersion;
    const fileInput = document.getElementById('import-file');
    const resultBox = document.getElementById('import-result');

    if (!lotteryId || !rulesetVersion || fileInput.files.length === 0) {
        alert('Selecciona la lotería, el archivo y un formato del sorteo.');
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('ruleset_version', rulesetVersion);

    resultBox.classList.remove('hidden');
    resultBox.textContent = 'Guardando la asignación y revisando el CSV...';

    try {
        const profile = await saveCsvSchemaProfile();
        const res = await fetch(`${API_BASE}/datasets/${lotteryId}/preview`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) {
            resultBox.textContent = [
                `Archivo: ${data.filename}`,
                `Filas: ${data.rows}`,
                `Fecha: ${profile.date_column || 'no indicada'}`,
                `Número/ID del sorteo: ${profile.event_id_column || 'no indicado'}`,
                ...Object.entries(profile.pool_columns).map(([pool, columns]) =>
                    `Números sorteados (${pool}): ${columns.join(', ')}`
                ),
                `Datos adicionales guardados: ${profile.allowed_attribute_columns.join(', ') || 'ninguno'}`,
                `Columnas ignoradas: ${profile.ignored_columns.join(', ') || 'ninguna'}`,
                'La asignación quedó guardada para esta lotería y este formato.'
            ].join('\n');
        } else {
            resultBox.textContent = `No se pudo revisar el archivo: ${data.detail || res.statusText}`;
        }
    } catch (err) {
        resultBox.textContent = `No se pudo revisar el archivo: ${err.message}`;
    }
}

// Importar CSV y Retreinar
async function uploadCsv() {
    const lotteryId = document.getElementById('import-lottery-select').value;
    const rulesetVersion = selectedRulesetVersion;
    const fileInput = document.getElementById('import-file');
    const resultBox = document.getElementById('import-result');

    if (!lotteryId || !rulesetVersion || fileInput.files.length === 0) {
        alert('Selecciona la lotería, el archivo y un formato del sorteo.');
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('ruleset_version', rulesetVersion);

    resultBox.classList.remove('hidden');
    resultBox.textContent = 'Guardando la asignación e importando el historial...';

    try {
        await saveCsvSchemaProfile();
        const res = await fetch(`${API_BASE}/datasets/${lotteryId}/import`, {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) {
            resultBox.textContent = JSON.stringify(data, null, 2);
            alert('¡Historial importado y modelo actualizado!');
        } else {
            resultBox.textContent = `No se pudo importar el historial: ${data.detail || res.statusText}`;
        }
    } catch (err) {
        resultBox.textContent = `Error de conexión al importar: ${err.message}`;
    }
}

// Consultar Próximo Sorteo
async function fetchNextDraw() {
    const lotteryId = document.getElementById('schedule-lottery-id').value;
    const resultBox = document.getElementById('schedule-result');
    if (!lotteryId) {
        alert('Ingrese un Lottery ID');
        return;
    }
    resultBox.classList.remove('hidden');
    resultBox.textContent = 'Consultando próximo sorteo...';
    try {
        const res = await fetch(`${API_BASE}/lotteries/${lotteryId}/next-draw`);
        const data = await res.json();
        if (res.ok) {
            resultBox.textContent = JSON.stringify(data, null, 2);
        } else {
            resultBox.textContent = `Error: ${JSON.stringify(data, null, 2)}`;
        }
    } catch (err) {
        resultBox.textContent = `Error de conexión: ${err.message}`;
    }
}

// Ejecutar Predicción
async function fetchPrediction() {
    const lotteryId = document.getElementById('predict-lottery-id').value;
    const resultBox = document.getElementById('prediction-result');
    if (!lotteryId) {
        alert('Ingrese un Lottery ID');
        return;
    }
    resultBox.classList.remove('hidden');
    resultBox.textContent = 'Calculando predicción y candidatos...';
    try {
        const res = await fetch(`${API_BASE}/lotteries/${lotteryId}/prediction`);
        const data = await res.json();
        if (res.ok) {
            resultBox.textContent = JSON.stringify(data, null, 2);
        } else {
            resultBox.textContent = `Error: ${JSON.stringify(data, null, 2)}`;
        }
    } catch (err) {
        resultBox.textContent = `Error de conexión: ${err.message}`;
    }
}

// Inicializar al cargar la página
window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('import-lottery-select').addEventListener('change', event => {
        if (event.target.value) {
            loadLotteryFormats(event.target.value);
        } else {
            resetFormatSelection();
        }
    });
    document.getElementById('import-file').addEventListener('change', inspectCsv);
    document.getElementById('format-existing-select').addEventListener('change', event => {
        selectSavedFormat(event.target.value);
    });
    document.getElementById('csv-column-roles').addEventListener('change', updateChronologyConfirmationVisibility);
    document.getElementById('format-new-button').addEventListener('click', () => {
        selectedRulesetVersion = null;
        document.getElementById('format-existing').classList.add('hidden');
        document.getElementById('format-current').classList.add('hidden');
        document.getElementById('format-setup').classList.remove('hidden');
        document.getElementById('import-actions').classList.add('hidden');
        if (!inspectedCsv) {
            document.getElementById('format-inspection').textContent =
                'Selecciona un CSV para recibir sugerencias; también puedes completar los valores manualmente.';
        }
    });
    document.getElementById('format-save-button').addEventListener('click', saveDrawFormat);
    document.getElementById('preview-csv-button').addEventListener('click', previewCsv);
    document.getElementById('upload-csv-button').addEventListener('click', uploadCsv);
    checkHealth();
});