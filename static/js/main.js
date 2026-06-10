// Global charts references
        const charts = {};
        let metadata = {};
        let mlMetrics = {};
        let trainingLogs = [];
        let walkthroughTimeout = null;

        window.addEventListener('DOMContentLoaded', () => {
            loadMetadata();
            loadMLMetrics();
            loadMLMetadata();
            loadAlerts();
            triggerUpdate();
            setupSmartLogListener();
        });

        // Load Filter Metadata
        async function loadMetadata() {
            try {
                const response = await fetch('/api/metadata');
                metadata = await response.json();
                
                // Populate filters
                populateDropdown(document.getElementById('global-region'), metadata.filters.regions);
                populateDropdown(document.getElementById('global-unit'), metadata.filters.units);
            } catch (e) {
                console.error("Error loading filter metadata:", e);
            }
        }

        function populateDropdown(selectElement, list) {
            if (!selectElement) return;
            selectElement.innerHTML = selectElement.tagName === 'SELECT' && selectElement.id.startsWith('global') ? selectElement.innerHTML : '';
            list.forEach(val => {
                if (val && val !== 'Unknown' && val !== 'UNKNOWN') {
                    const opt = document.createElement('option');
                    opt.value = val;
                    opt.textContent = val;
                    selectElement.appendChild(opt);
                }
            });
        }

        // Load ML Metrics
        async function loadMLMetrics() {
            try {
                const response = await fetch('/api/metrics');
                if (response.status === 200) {
                    mlMetrics = await response.json();
                    document.getElementById('walkthrough-status-banner').style.display = 'flex';
                    renderDeployedModelInfo();
                }
            } catch (e) {
                console.error("Error loading ML metrics:", e);
            }
        }

        // Switch active View
        function switchView(viewId, element) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
            
            document.querySelectorAll('.dashboard-view').forEach(view => view.classList.remove('active'));
            document.getElementById(`view-${viewId}`).classList.add('active');

            // Scroll workspace back to top on every view switch
            const ws = document.querySelector('.workspace');
            if (ws) ws.scrollTop = 0;
            
            // If walkthrough view and metrics loaded, run walkthrough animation
            if (viewId === 'model-report' && mlMetrics && trainingLogs.length === 0) {
                runWalkthroughAnimation();
            }

            // Resize NLP pane after scroll settles
            if (viewId === 'nlp-search') {
                // Double rAF ensures layout is fully painted after scroll reset
                requestAnimationFrame(() => requestAnimationFrame(resizeNlpPane));
            }
        }

        // Dynamically size the NLP two-col grid so both panels fill from their top to viewport bottom
        function resizeNlpPane() {
            const twoCol = document.getElementById('nlp-two-col');
            if (!twoCol) return;
            // getBoundingClientRect is relative to viewport — workspace.scrollTop must be 0
            const rect = twoCol.getBoundingClientRect();
            const available = window.innerHeight - rect.top - 32; // 32px bottom breathing room
            // Minimum 750px height ensures at least 4 complaint cards are visible at a time
            twoCol.style.height = Math.max(750, available) + 'px';
        }

        window.addEventListener('resize', () => {
            const nlpView = document.getElementById('view-nlp-search');
            if (nlpView && nlpView.classList.contains('active')) {
                requestAnimationFrame(resizeNlpPane);
            }
        });

        function updateSliders() {
            const startSlider = document.getElementById('start-year-slider');
            const endSlider = document.getElementById('end-year-slider');
            
            let start = parseInt(startSlider.value);
            let end = parseInt(endSlider.value);
            
            if (start > end) {
                // cross over prevention
                startSlider.value = end;
                start = end;
            }
            
            document.getElementById('start-year-lbl').innerText = start;
            document.getElementById('end-year-lbl').innerText = end;
            
            // Trigger dynamic update on trends
            triggerUpdate();
        }

        // Trigger dynamic API update for trends
        async function triggerUpdate() {
            const startYear = parseInt(document.getElementById('start-year-slider').value);
            const endYear = parseInt(document.getElementById('end-year-slider').value);
            const region = document.getElementById('global-region').value;
            const unit = document.getElementById('global-unit').value;
            
            const payload = {
                start_year: startYear,
                end_year: endYear,
                region: region,
                unit: unit
            };
            
            try {
                const response = await fetch('/api/trends', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                // Update KPIs
                document.getElementById('kpi-total').innerText = data.kpis.total_complaints.toLocaleString();
                document.getElementById('kpi-resolution').innerText = `${data.kpis.avg_resolution_days} Days`;
                document.getElementById('kpi-repetitive').innerText = `${data.kpis.pct_repetitive}%`;
                document.getElementById('kpi-severity').innerText = data.kpis.avg_severity;
                document.getElementById('kpi-debits').innerText = `₹ ${data.kpis.total_cost_debitable.toLocaleString()}`;
                
                // Update Charts
                renderAllCharts(data);
                
                // Fetch Equipment Analysis
                loadEquipmentAnalysis(payload);
            } catch (e) {
                console.error("Trends retrieval failed:", e);
            }
        }

        async function loadEquipmentAnalysis(payload) {
            try {
                const response = await fetch('/api/equipment_analysis', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                if (!data.equipment_data || data.equipment_data.length === 0) {
                    return;
                }
                
                const equipmentData = data.equipment_data;
                
                // Render Chart
                if (charts['equipmentBarChart']) {
                    charts['equipmentBarChart'].destroy();
                }
                const ctx = document.getElementById('equipmentBarChart').getContext('2d');
                charts['equipmentBarChart'] = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: equipmentData.map(d => d.product_name),
                        datasets: [{
                            label: 'Total Complaints',
                            data: equipmentData.map(d => d.total_complaints),
                            backgroundColor: '#ef4444',
                            borderRadius: 4
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
                
                // Render Playbook
                const container = document.getElementById('resolution-playbook-container');
                container.innerHTML = '';
                
                equipmentData.forEach(eq => {
                    const card = document.createElement('div');
                    card.style.border = "1px solid var(--border-color)";
                    card.style.borderRadius = "0.5rem";
                    card.style.padding = "1.25rem";
                    card.style.background = "#f8fafc";
                    
                    let defectsHtml = '';
                    eq.defects.forEach(def => {
                        defectsHtml += `
                            <div style="margin-top:1rem; border-top:1px solid #e2e8f0; padding-top:0.75rem;">
                                <div style="font-weight:700; color:var(--text-main); font-size:0.9rem;">Defect: ${def.defect_name} (Count: ${def.count})</div>
                                <div style="margin-top:0.5rem;">
                                    <span style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Historical Resolution:</span>
                                    <p style="font-size:0.85rem; color:var(--text-main); margin-top:0.25rem;">${def.disposition}</p>
                                </div>
                                <div style="margin-top:0.5rem;">
                                    <span style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Key Learnings / Preventive Measures:</span>
                                    <div class="learning-note-card" style="margin-top:0.25rem;">${def.learning}</div>
                                </div>
                            </div>
                        `;
                    });
                    
                    card.innerHTML = `
                        <h4 style="color:var(--bhel-blue); font-size:1rem; font-weight:700;">${eq.product_name}</h4>
                        <p style="font-size:0.8rem; color:var(--text-muted); font-weight:500;">Total Complaints: ${eq.total_complaints}</p>
                        ${defectsHtml}
                    `;
                    container.appendChild(card);
                });
                
            } catch (e) {
                console.error("Equipment Analysis retrieval failed:", e);
            }
        }

        // Render Charts using dynamic data
        function renderAllCharts(data) {
            // Destroy all existing charts to avoid overlap glitches
            Object.keys(charts).forEach(key => {
                if (charts[key]) charts[key].destroy();
            });
            
            if (data.empty) return;
            
            // 1. Complaint Volume Trend (YoY Line chart)
            const volCtx = document.getElementById('chart-vol').getContext('2d');
            const volDatasets = Object.keys(data.complaint_volume.yoy_volume).map((yr, idx) => {
                const colors = ['#003594', '#F7941D', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#f59e0b', '#06b6d4', '#14b8a6', '#84cc16'];
                return {
                    label: yr,
                    data: data.complaint_volume.yoy_volume[yr],
                    borderColor: colors[idx % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.25
                };
            });
            charts['chart-vol'] = new Chart(volCtx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: volDatasets
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 2. Defect Type Overview (Bar)
            const defCtx = document.getElementById('chart-defect-type-overview').getContext('2d');
            const defKeys = Object.keys(data.defect_analysis.top_defect_types);
            charts['chart-defect-type-overview'] = new Chart(defCtx, {
                type: 'bar',
                data: {
                    labels: defKeys,
                    datasets: [{
                        label: 'Defect Counts',
                        data: defKeys.map(k => data.defect_analysis.top_defect_types[k]),
                        backgroundColor: '#003594',
                        borderColor: '#002366',
                        borderWidth: 1
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
            });
            
            // 3. Average SLA Overview (Line)
            const slaCtx = document.getElementById('chart-sla-overview').getContext('2d');
            const slaYears = Object.keys(data.resolution_performance.avg_days_year);
            charts['chart-sla-overview'] = new Chart(slaCtx, {
                type: 'line',
                data: {
                    labels: slaYears,
                    datasets: [{
                        label: 'Avg Resolution Lag (Days)',
                        data: slaYears.map(y => data.resolution_performance.avg_days_year[y]),
                        borderColor: '#F7941D',
                        backgroundColor: 'rgba(247,148,29,0.05)',
                        fill: true,
                        tension: 0.1,
                        borderWidth: 3
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 4. Rolling 3-month volume (Line)
            const rollCtx = document.getElementById('chart-rolling').getContext('2d');
            charts['chart-rolling'] = new Chart(rollCtx, {
                type: 'line',
                data: {
                    labels: data.complaint_volume.rolling_labels,
                    datasets: [{
                        label: '3-Month Rolling Average Count',
                        data: data.complaint_volume.rolling_3,
                        borderColor: '#003594',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.2
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 5. Complaint Status Area
            const statusCtx = document.getElementById('chart-status-dist').getContext('2d');
            const statusLabels = data.milestone_status.status_data.years;
            const statusKeys = data.milestone_status.status_data.statuses;
            const statusDatasets = statusKeys.map((k, idx) => {
                const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#64748b'];
                return {
                    label: k,
                    data: data.milestone_status.status_data.series[k],
                    backgroundColor: colors[idx % colors.length] + 'aa',
                    borderColor: colors[idx % colors.length],
                    fill: true
                };
            });
            charts['chart-status-dist'] = new Chart(statusCtx, {
                type: 'line',
                data: { labels: statusLabels, datasets: statusDatasets },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { stacked: true } } }
            });
            
            // 6. Top 10 Sub-types (Bar)
            const subCtx = document.getElementById('chart-defect-subtypes').getContext('2d');
            const subKeys = Object.keys(data.defect_analysis.top_defect_subtypes);
            charts['chart-defect-subtypes'] = new Chart(subCtx, {
                type: 'bar',
                data: {
                    labels: subKeys.map(k => k.length > 25 ? k.substring(0, 25) + '...' : k),
                    datasets: [{
                        label: 'Count',
                        data: subKeys.map(k => data.defect_analysis.top_defect_subtypes[k]),
                        backgroundColor: '#F7941D'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 7. Defect by Region (Grouped Bar)
            const defRegCtx = document.getElementById('chart-defect-region').getContext('2d');
            const regLabels = data.defect_analysis.defect_region.regions;
            const regDefects = Object.keys(data.defect_analysis.defect_region.defects);
            const regDatasets = regDefects.map((def, idx) => {
                const colors = ['#003594', '#F7941D', '#10b981', '#3b82f6', '#8b5cf6'];
                return {
                    label: def,
                    data: data.defect_analysis.defect_region.defects[def],
                    backgroundColor: colors[idx % colors.length]
                };
            });
            charts['chart-defect-region'] = new Chart(defRegCtx, {
                type: 'bar',
                data: { labels: regLabels, datasets: regDatasets },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 8. Resolution bins
            const resBinsCtx = document.getElementById('chart-res-bins').getContext('2d');
            const binKeys = Object.keys(data.resolution_performance.resolution_bins);
            charts['chart-res-bins'] = new Chart(resBinsCtx, {
                type: 'bar',
                data: {
                    labels: binKeys,
                    datasets: [{
                        label: 'Complaints count',
                        data: binKeys.map(k => data.resolution_performance.resolution_bins[k]),
                        backgroundColor: '#003594'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 9. SLA ratios frame (Doughnut)
            const ratioCtx = document.getElementById('chart-sla-ratios').getContext('2d');
            charts['chart-sla-ratios'] = new Chart(ratioCtx, {
                type: 'bar',
                data: {
                    labels: ['Within 30 Days', 'Within 60 Days', 'Within 90 Days'],
                    datasets: [{
                        label: 'Percentage (%) of resolved complaints',
                        data: [
                            data.resolution_performance.resolved_ratios.within_30,
                            data.resolution_performance.resolved_ratios.within_60,
                            data.resolution_performance.resolved_ratios.within_90
                        ],
                        backgroundColor: ['#10b981', '#3b82f6', '#f59e0b']
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
            });
            
            // 10. Longest Units
            const longUnitCtx = document.getElementById('chart-longest-units').getContext('2d');
            const luKeys = Object.keys(data.resolution_performance.longest_resolution_units);
            charts['chart-longest-units'] = new Chart(longUnitCtx, {
                type: 'bar',
                data: {
                    labels: luKeys,
                    datasets: [{
                        label: 'Average Resolution Time (Days)',
                        data: luKeys.map(k => data.resolution_performance.longest_resolution_units[k]),
                        backgroundColor: '#ef4444'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 11. Cost claimed vs accepted
            const costCtx = document.getElementById('chart-cost-claimed-accepted').getContext('2d');
            const costYears = Object.keys(data.cost_analysis.avg_claimed_year);
            charts['chart-cost-claimed-accepted'] = new Chart(costCtx, {
                type: 'line',
                data: {
                    labels: costYears,
                    datasets: [
                        {
                            label: 'Avg Claimed (INR)',
                            data: costYears.map(y => data.cost_analysis.avg_claimed_year[y]),
                            borderColor: '#003594',
                            backgroundColor: 'transparent',
                            borderWidth: 2
                        },
                        {
                            label: 'Avg Accepted (INR)',
                            data: costYears.map(y => data.cost_analysis.avg_accepted_year[y]),
                            borderColor: '#10b981',
                            backgroundColor: 'transparent',
                            borderWidth: 2
                        }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 12. Cost overrun trend
            const overCtx = document.getElementById('chart-overrun').getContext('2d');
            const overYears = Object.keys(data.cost_analysis.avg_overrun_year);
            charts['chart-overrun'] = new Chart(overCtx, {
                type: 'bar',
                data: {
                    labels: overYears,
                    datasets: [{
                        label: 'Avg Overrun Cost (INR)',
                        data: overYears.map(y => data.cost_analysis.avg_overrun_year[y]),
                        backgroundColor: '#F7941D'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
            
            // 13. Vendor costs
            const vendCtx = document.getElementById('chart-vendor-costs').getContext('2d');
            const vendKeys = Object.keys(data.cost_analysis.top_vendors_cost);
            charts['chart-vendor-costs'] = new Chart(vendCtx, {
                type: 'bar',
                data: {
                    labels: vendKeys,
                    datasets: [{
                        label: 'Total Cost (INR)',
                        data: vendKeys.map(k => data.cost_analysis.top_vendors_cost[k]),
                        backgroundColor: '#003594'
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            // 14. RCA - NC Categorization
            if (data.rca_global && data.rca_global.top_nc_types) {
                const ncCtx = document.getElementById('chart-rca-nc').getContext('2d');
                const ncKeys = Object.keys(data.rca_global.top_nc_types);
                charts['chart-rca-nc'] = new Chart(ncCtx, {
                    type: 'bar',
                    data: {
                        labels: ncKeys.map(k => k.length > 25 ? k.substring(0, 25) + '...' : k),
                        datasets: [{
                            label: 'NC Counts',
                            data: ncKeys.map(k => data.rca_global.top_nc_types[k]),
                            backgroundColor: '#003594',
                            borderWidth: 1
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
                });
            }

            // 15. RCA - Defect by Product (Stacked Bar)
            if (data.rca_global && data.rca_global.defect_product) {
                const dpCtx = document.getElementById('chart-rca-defect-product').getContext('2d');
                const dpProducts = data.rca_global.defect_product.products;
                const dpDefects = Object.keys(data.rca_global.defect_product.defects);
                const colors = ['#003594', '#F7941D', '#10b981', '#3b82f6', '#8b5cf6'];
                const dpDatasets = dpDefects.map((def, idx) => ({
                    label: def,
                    data: data.rca_global.defect_product.defects[def],
                    backgroundColor: colors[idx % colors.length]
                }));
                charts['chart-rca-defect-product'] = new Chart(dpCtx, {
                    type: 'bar',
                    data: {
                        labels: dpProducts,
                        datasets: dpDatasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { x: { stacked: true }, y: { stacked: true } }
                    }
                });
            }

            // 16. RCA - Learnings List
            const learningsList = document.getElementById('rca-learnings-list');
            if (learningsList && data.rca_global && data.rca_global.learnings) {
                learningsList.innerHTML = '';
                if (data.rca_global.learnings.length === 0) {
                    learningsList.innerHTML = '<div style="color:var(--text-muted); font-style:italic; padding:0.5rem 0;">No learning data recorded in selected dataset subset.</div>';
                } else {
                    data.rca_global.learnings.forEach(lrn => {
                        const item = document.createElement('div');
                        item.style.display = 'flex';
                        item.style.gap = '0.5rem';
                        item.style.alignItems = 'flex-start';
                        item.style.background = 'rgba(0,53,148,0.02)';
                        item.style.padding = '0.75rem';
                        item.style.borderRadius = '0.5rem';
                        item.style.borderLeft = '3px solid var(--bhel-blue)';
                        
                        item.innerHTML = `
                            <svg width="16" height="16" fill="none" stroke="var(--bhel-blue)" stroke-width="2.5" viewBox="0 0 24 24" style="margin-top:2px; flex-shrink: 0;"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            <span>${lrn}</span>
                        `;
                        learningsList.appendChild(item);
                    });
                }
            }
        }

        // Export active Chart as Image
        function exportChart(chartId) {
            const chart = charts[chartId];
            if (chart) {
                const url = chart.toBase64Image();
                const a = document.createElement('a');
                a.href = url;
                a.download = `${chartId}_export.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        }

        // Export Leaderboard Table as CSV
        function exportLeaderboardCSV() {
            let csv = "Target,Algorithm,Metric A (Accuracy/MAE),Metric B (Precision/RMSE),Metric C (Recall/R2),Metric D (F1-score),Status\n";
            const rows = document.querySelectorAll("#leaderboard-tbody tr");
            rows.forEach(row => {
                const cols = row.querySelectorAll("td");
                const rowData = Array.from(cols).map(c => `"${c.innerText}"`).join(",");
                csv += rowData + "\n";
            });
            
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "BHEL_ML_Leaderboard_Report.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }

        // 10x ML Engine State
        let mlModelsMetadata = {};
        let selectedMLTargets = new Set();
        
        async function loadMLMetadata() {
            try {
                const response = await fetch('/api/ml_metadata');
                const data = await response.json();
                if (data.status === 'success') {
                    mlModelsMetadata = data.models;
                    renderMLTargetsGrid();
                }
            } catch (e) {
                console.error("Error loading ML metadata:", e);
            }
        }
        
        function renderMLTargetsGrid() {
            const grid = document.getElementById('ml-targets-grid');
            grid.innerHTML = '';
            
            const descriptions = {
                'Resolution_Time': 'Predicts the number of days a complaint will take to reach final disposition.',
                'Severity': 'Predicts the severity rating given by the unit.',
                'Cost': 'Estimates the likely financial expenditure incurred to resolve the issue.',
                'Repeat_Failure': 'Identifies if the complaint is highly likely to become a recurring issue.',
                'Vendor_Risk': 'Calculates a risk score indicating the likelihood of delays and extra costs from the vendor.',
                'Defect_Root_Cause': 'Uses NLP on text descriptions to predict the likely defect category.',
                'Escalation': 'Predicts the probability that this complaint will be reopened.',
                'Delay': 'Predicts whether the resolution time will exceed the 15-day SLA.',
                'Warranty_Recovery': 'Predicts the probability of successfully recovering debit costs from the vendor.',
                'Reliability': 'Scores the equipment/vendor on a 0-100 reliability index based on predicted risk metrics.'
            };
            
            Object.keys(mlModelsMetadata).forEach(taskName => {
                const card = document.createElement('div');
                card.className = 'prediction-target-card';
                card.style.cssText = `
                    border: 1px solid var(--border-color);
                    border-radius: 0.5rem;
                    padding: 1rem;
                    cursor: pointer;
                    transition: all 0.2s;
                    background: var(--card-bg);
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                `;
                
                const readableName = taskName.replace(/_/g, ' ');
                const desc = descriptions[taskName] || 'Machine learning model predicting historical metrics.';
                
                card.innerHTML = `
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <h4 style="margin:0 0 0.5rem 0; font-size: 0.95rem; font-weight: 600;">${readableName}</h4>
                            <input type="checkbox" id="ml-chk-${taskName}" style="pointer-events:none; margin-top:0.25rem;">
                        </div>
                        <p style="font-size: 0.75rem; color: var(--text-muted); margin: 0; line-height: 1.4;">${desc}</p>
                    </div>
                `;
                
                card.onclick = () => {
                    const chk = document.getElementById(`ml-chk-${taskName}`);
                    if (selectedMLTargets.has(taskName)) {
                        selectedMLTargets.delete(taskName);
                        chk.checked = false;
                        card.style.borderColor = 'var(--border-color)';
                        card.style.background = 'var(--card-bg)';
                    } else {
                        selectedMLTargets.add(taskName);
                        chk.checked = true;
                        card.style.borderColor = 'var(--bhel-blue)';
                        card.style.background = 'var(--primary-glow)';
                    }
                    updateDynamicFeatureForm();
                };
                
                grid.appendChild(card);
            });
        }
        
        function updateDynamicFeatureForm() {
            const section = document.getElementById('ml-features-section');
            const grid = document.getElementById('ml-dynamic-form-grid');
            
            if (selectedMLTargets.size === 0) {
                section.style.display = 'none';
                document.getElementById('prediction-result-panel').style.display = 'none';
                return;
            }
            
            section.style.display = 'block';
            grid.innerHTML = '';
            
            let requiredFeatures = new Set();
            let requiresNlp = false;
            
            selectedMLTargets.forEach(task => {
                const model = mlModelsMetadata[task];
                if (model.is_nlp) {
                    requiresNlp = true;
                } else {
                    model.features.forEach(f => requiredFeatures.add(f));
                }
            });
            
            if (requiresNlp) {
                const fg = document.createElement('div');
                fg.className = 'form-group span-2';
                fg.innerHTML = `
                    <label>Problem Description (Text)</label>
                    <textarea id="pred-feat-Problem_Description" class="form-input" required style="min-height:80px; width: 100%;" placeholder="Describe the defect or problem..."></textarea>
                `;
                grid.appendChild(fg);
            }
            
            requiredFeatures.forEach(feat => {
                const fg = document.createElement('div');
                fg.className = 'form-group';
                const readableFeat = feat.replace(/_/g, ' ');
                
                let mappings = [];
                for (let task of selectedMLTargets) {
                    if (mlModelsMetadata[task] && mlModelsMetadata[task].label_mappings && mlModelsMetadata[task].label_mappings[feat]) {
                        mappings = mlModelsMetadata[task].label_mappings[feat];
                        break;
                    }
                }
                
                if (mappings.length > 0) {
                    let opts = mappings.map(m => `<option value="${m}">${m}</option>`).join('');
                    fg.innerHTML = `
                        <label>${readableFeat}</label>
                        <select id="pred-feat-${feat}" class="form-input" required>
                            <option value="">Select...</option>
                            ${opts}
                        </select>
                    `;
                } else {
                    fg.innerHTML = `
                        <label>${readableFeat}</label>
                        <input type="text" id="pred-feat-${feat}" class="form-input" required placeholder="Enter value...">
                    `;
                }
                
                grid.appendChild(fg);
            });
        }
        
        async function runPrediction(e) {
            e.preventDefault();
            
            if (selectedMLTargets.size === 0) return;
            
            const submitBtn = document.getElementById('predict-submit-btn');
            submitBtn.disabled = true;
            submitBtn.innerHTML = `<div class="spinner"></div> Running Inference...`;
            
            const payloadFeatures = {};
            const gridInputs = document.querySelectorAll('#ml-dynamic-form-grid .form-input');
            gridInputs.forEach(input => {
                const featName = input.id.replace('pred-feat-', '');
                payloadFeatures[featName] = input.value;
            });
            
            const payload = {
                targets: Array.from(selectedMLTargets),
                features: payloadFeatures
            };
            
            try {
                const response = await fetch('/api/predict_multiple', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const res = await response.json();
                
                if (res.status === 'success') {
                    renderMLResults(res.predictions);
                } else {
                    alert("Error: " + res.error);
                }
            } catch (err) {
                console.error("Prediction inference failed", err);
                alert("Error during model inference.");
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = "Run Predictions";
            }
        }
        
        function renderMLResults(predictions) {
            const panel = document.getElementById('prediction-result-panel');
            panel.style.display = 'flex';
            panel.innerHTML = '';
            
            Object.keys(predictions).forEach(task => {
                const result = predictions[task];
                const readableName = task.replace(/_/g, ' ');
                
                const card = document.createElement('div');
                card.className = 'prediction-outcome-card';
                card.style.flex = '1 1 250px';
                
                let valColor = 'var(--text-main)';
                let extraHtml = '';
                let extraClass = '';
                
                if (result.error) {
                    valColor = 'var(--danger)';
                    extraHtml = `<span style="color:var(--danger)">${result.error}</span>`;
                } else {
                    if (result.type === 'classification') {
                        extraHtml += `
                            <div style="margin-top:0.5rem;">
                                <span style="font-size:0.8rem; color:var(--text-muted);">Confidence Score</span>
                                <div class="confidence-bar-bg">
                                    <div class="confidence-bar-fill" style="width:${result.confidence}%;">${result.confidence}%</div>
                                </div>
                            </div>
                        `;
                    }
                    
                    // Vendor Lead Time
                    if (result.vendor_lead_time) {
                        extraHtml += `
                            <div style="margin-top:0.5rem; font-size:0.8rem; color:var(--text-muted);">
                                <strong>Vendor Note:</strong> This vendor historically takes ${result.vendor_lead_time} days.
                            </div>
                        `;
                    }
                    
                    // Risk Warnings
                    if (task === 'Resolution_Time' && result.prediction && parseFloat(result.prediction) > 90) {
                        extraClass = 'flash-warning';
                        extraHtml += `
                            <div style="margin-top:0.5rem; font-size:0.8rem; color:var(--danger); font-weight:bold;">
                                🚨 SLA BREACH RISK - Penalty Likely
                            </div>
                        `;
                    }
                    
                    // Check for Severity if Cost Debitable is Y (but we don't have cost debitable easily accessible here, so we will trigger overrun warning on Cost > 100k)
                    if (task === 'Cost' && result.prediction && parseFloat(result.prediction.toString().replace(/,/g, '')) > 100000) {
                        extraClass = 'flash-warning';
                        extraHtml += `
                            <div style="margin-top:0.5rem; font-size:0.8rem; color:var(--danger); font-weight:bold;">
                                🚨 HIGH COST OVERRUN RISK
                            </div>
                        `;
                    }
                }
                
                if (extraClass) card.classList.add(extraClass);
                
                const predVal = result.prediction || '-';
                
                card.innerHTML = `
                    <span class="label" style="font-weight:600; color:var(--bhel-blue);">${readableName}</span>
                    <span class="val" style="color: ${valColor}; font-size: 1.5rem; display:block; margin: 0.5rem 0;">${predVal}</span>
                    ${extraHtml}
                `;
                panel.appendChild(card);
            });
        }

        // Deployed Model Information & Plots
        function renderDeployedModelInfo() {
            // Target A
            document.getElementById('deployed-severity-name').innerText = mlMetrics.targets.severity.best_model;
            const bestSevObj = mlMetrics.targets.severity.results.find(r => r.model_name === mlMetrics.targets.severity.best_model);
            document.getElementById('deployed-severity-metrics').innerText = `Accuracy: ${bestSevObj.accuracy.toFixed(4)} | F1-Score: ${bestSevObj.f1_score.toFixed(4)}`;
            renderImportancePlot('chart-imp-severity', bestSevObj.feature_importances, '#003594');
            
            // Target B
            document.getElementById('deployed-disposition-name').innerText = mlMetrics.targets.disposition.best_model;
            const bestDispObj = mlMetrics.targets.disposition.results.find(r => r.model_name === mlMetrics.targets.disposition.best_model);
            document.getElementById('deployed-disposition-metrics').innerText = `RMSE: ${bestDispObj.rmse.toFixed(1)} Days | MAE: ${bestDispObj.mae.toFixed(1)} Days | R²: ${bestDispObj.r2_score.toFixed(2)}`;
            renderScatterPlot('chart-scatter-disposition', bestDispObj.scatter_plot_sample);
            
            // Target C
            document.getElementById('deployed-repetitive-name').innerText = mlMetrics.targets.repetitive.best_model;
            const bestRepObj = mlMetrics.targets.repetitive.results.find(r => r.model_name === mlMetrics.targets.repetitive.best_model);
            document.getElementById('deployed-repetitive-metrics').innerText = `Accuracy: ${bestRepObj.accuracy.toFixed(4)} | F1-Score: ${bestRepObj.f1_score.toFixed(4)}`;
            renderImportancePlot('chart-imp-repetitive', bestRepObj.feature_importances, '#F7941D');
        }

        function renderImportancePlot(canvasId, importances, color) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            const labels = importances.map(i => i.feature);
            const vals = importances.map(i => i.importance);
            charts[canvasId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feature Importance Score',
                        data: vals,
                        backgroundColor: color
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, indexAxis: 'y' }
            });
        }

        function renderScatterPlot(canvasId, scatterData) {
            const ctx = document.getElementById(canvasId).getContext('2d');
            const dataPoints = scatterData.map(d => ({ x: d.actual, y: d.predicted }));
            charts[canvasId] = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Actual vs Predicted SLA Days',
                        data: dataPoints,
                        backgroundColor: 'rgba(0, 53, 148, 0.4)',
                        borderColor: '#003594',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Actual Days' } },
                        y: { title: { display: true, text: 'Predicted Days' } }
                    }
                }
            });
        }

        // Training Walkthrough effect
        function runWalkthroughAnimation() {
            const banner = document.getElementById('walkthrough-status-banner');
            const bannerText = document.getElementById('walkthrough-banner-text');
            const tbody = document.getElementById('leaderboard-tbody');
            
            banner.style.display = 'flex';
            tbody.innerHTML = '';
            
            if (walkthroughTimeout) clearTimeout(walkthroughTimeout);
            
            // Build logging steps
            trainingLogs = [];
            
            // Group models by target
            const addTargetLogs = (targetKey, targetName, metricKeys) => {
                const targetObj = mlMetrics.targets[targetKey];
                targetObj.results.forEach(m => {
                    const isBest = m.model_name === targetObj.best_model;
                    const rowHtml = `
                        <tr class="${isBest ? 'green-highlight' : ''}">
                            <td style="font-weight:600;">${targetName}</td>
                            <td>${m.model_name}</td>
                            <td>${metricKeys[0]}: ${(m[metricKeys[0]] || m.classification_report.macro_avg['precision'] || 0).toFixed(4)}</td>
                            <td>${metricKeys[1]}: ${(m[metricKeys[1]] || m.classification_report.macro_avg['recall'] || 0).toFixed(4)}</td>
                            <td>${metricKeys[2]}: ${(m[metricKeys[2]] || m.classification_report.macro_avg['f1-score'] || 0).toFixed(4)}</td>
                            <td>${metricKeys[3]}: ${(m[metricKeys[3]] || 0).toFixed(4)}</td>
                            <td><span style="color: ${isBest ? 'var(--success)' : 'var(--text-muted)'}; font-weight:${isBest?700:500};">${isBest ? 'DEPLOYED ✓' : 'Evaluated'}</span></td>
                        </tr>
                    `;
                    trainingLogs.push({
                        text: `Evaluating Target [${targetName}] using [${m.model_name}] algorithm...`,
                        rowHtml: rowHtml
                    });
                });
            };
            
            addTargetLogs('severity', 'Severity Rating', ['accuracy', 'precision', 'recall', 'f1_score']);
            addTargetLogs('disposition', 'Disposition SLA', ['mae', 'rmse', 'r2_score', 'mae']);
            addTargetLogs('repetitive', 'Repetitive Risk', ['accuracy', 'precision', 'recall', 'f1_score']);
            
            let currentStep = 0;
            
            function stepAnimation() {
                if (currentStep < trainingLogs.length) {
                    bannerText.innerHTML = `<strong>Training Walkthrough:</strong> ${trainingLogs[currentStep].text}`;
                    tbody.innerHTML += trainingLogs[currentStep].rowHtml;
                    currentStep++;
                    walkthroughTimeout = setTimeout(stepAnimation, 300); // 300ms delay per model training simulation
                } else {
                    bannerText.innerHTML = `<strong>Walkthrough Completed!</strong> Trained all 21 models (7 models &times; 3 targets). Highlights indicate best-performing algorithms deployed.`;
                }
            }
            
            stepAnimation();
        }

        // Apply tag click suggestion
        function applySuggestion(text) {
            document.getElementById('nlp-search-query').value = text;
            executeNlpSearch();
        }

        // Semantics NLP search execution
        async function executeNlpSearch() {
            const query = document.getElementById('nlp-search-query').value.trim();
            const startYear = parseInt(document.getElementById('start-year-slider').value);
            const endYear = parseInt(document.getElementById('end-year-slider').value);
            const region = document.getElementById('global-region').value;
            const unit = document.getElementById('global-unit').value;
            
            const resultsList = document.getElementById('nlp-results-list');
            const resultsCount = document.getElementById('nlp-results-count');
            const rcaSummary = document.getElementById('nlp-rca-summary');
            
            // Show loading
            resultsList.innerHTML = `
                <div style="display:flex; justify-content:center; align-items:center; padding:4rem; flex-direction:column; gap:1rem;">
                    <div class="spinner"></div>
                    <div style="color:var(--text-muted); font-size:0.9rem;">Searching & analyzing text semantics...</div>
                </div>
            `;
            
            const payload = {
                query: query,
                start_year: startYear,
                end_year: endYear,
                region: region,
                unit: unit,
                limit: 50
            };
            
            try {
                const response = await fetch('/api/nlp_search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultsList.innerHTML = `<div style="color:var(--danger); padding:2rem; text-align:center;">Error: ${data.error}</div>`;
                    return;
                }
                
                resultsCount.innerText = data.total_results.toLocaleString();
                
                // 1. Populate RCA Summary with progress bars
                if (data.rca_summary) {
                    const r = data.rca_summary;
                    
                    // Helper to build progress bars
                    const buildProgressBarHtml = (obj, colorClass) => {
                        const entries = Object.entries(obj);
                        if (entries.length === 0) return '<div style="color:var(--text-muted); font-size:0.8rem; font-style:italic;">None</div>';
                        const maxVal = Math.max(...entries.map(e => e[1]));
                        return `<div class="progress-list">` + entries.map(([key, val]) => {
                            const pct = maxVal > 0 ? (val / maxVal * 100) : 0;
                            return `
                                <div class="progress-item">
                                    <div class="progress-item-label">
                                        <span style="color:var(--text-main);">${key}</span>
                                        <span style="color:var(--text-muted);">${val}</span>
                                    </div>
                                    <div class="progress-bar-bg">
                                        <div class="progress-bar-fill ${colorClass}" style="width: ${pct}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('') + `</div>`;
                    };
                    
                    let defectsHtml = buildProgressBarHtml(r.top_defects, 'orange');
                    let ncHtml = buildProgressBarHtml(r.top_nc, '');
                    let vendorHtml = buildProgressBarHtml(r.top_vendors, 'green');
                    
                    rcaSummary.innerHTML = `
                        <div style="display:flex; flex-direction:column; gap:1.25rem;">
                            <!-- Semantic KPIs -->
                            <div class="kpi-mini-grid">
                                <div class="kpi-mini-card">
                                    <span class="kpi-mini-label">Avg Resolution</span>
                                    <span class="kpi-mini-val">${r.avg_resolution_days} <span style="font-size:0.75rem; font-weight:600; color:var(--text-muted);">Days</span></span>
                                </div>
                                <div class="kpi-mini-card">
                                    <span class="kpi-mini-label">Avg Severity</span>
                                    <span class="kpi-mini-val">${r.avg_severity}</span>
                                </div>
                            </div>
                            
                            <!-- Top Defect Types -->
                            <div style="display:flex; flex-direction:column; gap:0.5rem;">
                                <div style="font-weight:700; color:var(--bhel-blue); font-size:0.75rem; text-transform:uppercase; border-bottom:1px solid var(--border-color); padding-bottom:0.25rem;">Top Defect Types</div>
                                ${defectsHtml}
                            </div>
                            
                            <!-- Top NC Categories -->
                            <div style="display:flex; flex-direction:column; gap:0.5rem;">
                                <div style="font-weight:700; color:var(--bhel-blue); font-size:0.75rem; text-transform:uppercase; border-bottom:1px solid var(--border-color); padding-bottom:0.25rem;">Top NC Categories</div>
                                ${ncHtml}
                            </div>

                            <!-- Top Vendor Sources -->
                            <div style="display:flex; flex-direction:column; gap:0.5rem;">
                                <div style="font-weight:700; color:var(--bhel-blue); font-size:0.75rem; text-transform:uppercase; border-bottom:1px solid var(--border-color); padding-bottom:0.25rem;">Top Vendor Sources</div>
                                ${vendorHtml}
                            </div>

                            <!-- Action Learnings -->
                            <div style="display:flex; flex-direction:column; gap:0.5rem;">
                                <div style="font-weight:700; color:var(--bhel-orange); font-size:0.75rem; text-transform:uppercase; border-bottom:1px solid var(--border-color); padding-bottom:0.25rem;">RCA Action Learnings</div>
                                <div style="display:flex; flex-direction:column; gap:0.5rem;">
                                    ${r.learnings.map(l => `<div class="learning-note-card">${l}</div>`).join('') || '<div style="color:var(--text-muted); font-size:0.8rem; font-style:italic;">No learning entries available.</div>'}
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                // 2. Populate Results list
                resultsList.innerHTML = '';
                if (data.results.length === 0) {
                    resultsList.innerHTML = `
                        <div class="empty-state" style="padding: 4rem 1rem;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:48px; height:48px; color:var(--text-muted); margin-bottom:1rem;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                            <h3 style="color:var(--text-muted);">No complaints matched your keywords.</h3>
                            <p style="font-size:0.85rem; color:var(--text-muted);">Try broadening your query or removing active filters.</p>
                        </div>
                    `;
                    return;
                }
                
                data.results.forEach(res => {
                    const card = document.createElement('div');
                    card.className = 'result-card-custom';
                    
                    // Assign border-left color depending on severity
                    const unitSev = parseFloat(res['Severity Rating (Given by Unit)'] || res['Severity Rating (Given by Site)'] || 0.5);
                    let borderLeftColor = 'var(--bhel-blue)';
                    let severityLabel = 'Medium';
                    let severityClass = 'orange';
                    if (unitSev >= 0.7) {
                        borderLeftColor = 'var(--danger)';
                        severityLabel = 'High Severity';
                        severityClass = 'orange';
                    } else if (unitSev <= 0.35) {
                        borderLeftColor = 'var(--success)';
                        severityLabel = 'Low Severity';
                        severityClass = 'green';
                    } else {
                        severityLabel = 'Medium Severity';
                        severityClass = 'blue';
                    }
                    card.style.borderLeft = `5px solid ${borderLeftColor}`;
                    
                    // Highlight query tokens
                    let desc = res['Problem Description'] || 'No description provided';
                    let rec = res['Site Recommendation'] || 'No site recommendation provided';
                    let learn = res['Learning Derived'] || 'No learning derived';
                    
                    if (query && data.query_tokens && data.query_tokens.length > 0) {
                        data.query_tokens.forEach(tok => {
                            if (tok && tok.length > 2) {
                                const regex = new RegExp(`(${tok})`, 'gi');
                                desc = desc.replace(regex, '<mark style="background:rgba(247,148,29,0.25); color:inherit; font-weight:600; padding:0 2px; border-radius:2px;">$1</mark>');
                                rec = rec.replace(regex, '<mark style="background:rgba(247,148,29,0.25); color:inherit; font-weight:600; padding:0 2px; border-radius:2px;">$1</mark>');
                                learn = learn.replace(regex, '<mark style="background:rgba(247,148,29,0.25); color:inherit; font-weight:600; padding:0 2px; border-radius:2px;">$1</mark>');
                            }
                        });
                    }
                    
                    const scoreBadge = query ? `
                        <span style="font-size:0.75rem; font-weight:700; color:var(--bhel-orange); background:rgba(247,148,29,0.08); padding:0.2rem 0.5rem; border-radius:0.25rem;">
                            Match: ${(res.score || 0).toFixed(3)}
                        </span>
                    ` : '';
                    
                    card.innerHTML = `
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem; border-bottom:1px solid rgba(0,0,0,0.03); padding-bottom:0.5rem; margin-bottom:0.25rem;">
                            <div>
                                <strong style="color:var(--bhel-blue); font-size:0.95rem;">${res['Complaint Number'] || 'Unknown No.'} (Sno: ${res['Sno'] || 'N/A'})</strong>
                                <span style="font-size:0.75rem; color:var(--text-muted); margin-left:0.5rem;">[${res['Complaint Date'] ? res['Complaint Date'].substring(0, 10) : 'N/A'}]</span>
                            </div>
                            <div style="display:flex; gap:0.4rem; align-items:center;">
                                ${scoreBadge}
                                <span class="result-pill ${res['Status'] === 'Closed' ? 'green' : 'blue'}">
                                    ${res['Status'] || 'Unknown'}
                                </span>
                            </div>
                        </div>
                        
                        <div style="font-size:0.88rem; color:var(--text-main); line-height:1.45;">
                            <strong style="color:var(--bhel-blue); font-size:0.75rem; text-transform:uppercase; display:block; margin-bottom:0.25rem; letter-spacing:0.02rem;">Problem Description</strong>
                            ${desc}
                        </div>

                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:0.75rem; margin-top:0.35rem;">
                            <div style="background:rgba(0,53,148,0.01); border: 1px solid rgba(0,53,148,0.03); padding:0.65rem; border-radius:0.35rem; font-size:0.8rem; line-height:1.4;">
                                <strong style="color:var(--text-muted); font-size:0.7rem; text-transform:uppercase; display:block; margin-bottom:0.25rem;">Site Recommendation</strong>
                                ${rec}
                            </div>
                            <div style="background:rgba(247,148,29,0.01); border: 1px solid rgba(247,148,29,0.03); padding:0.65rem; border-radius:0.35rem; font-size:0.8rem; line-height:1.4;">
                                <strong style="color:var(--text-muted); font-size:0.7rem; text-transform:uppercase; display:block; margin-bottom:0.25rem;">Learning Derived</strong>
                                ${learn}
                            </div>
                        </div>

                        <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; padding-top:0.5rem; border-top:1px solid rgba(0,0,0,0.03);">
                            <span class="result-pill blue">Product: ${res['Product'] || 'N/A'}</span>
                            <span class="result-pill">Project: ${res['Project'] || 'N/A'}</span>
                            <span class="result-pill">Defect: ${res['Defect Type'] || 'N/A'}</span>
                            <span class="result-pill">NC Cat: ${res['NC Categorization'] || 'N/A'}</span>
                            <span class="result-pill">SLA: ${res['Days Taken for Disposition'] || 'N/A'} Days</span>
                            <span class="result-pill ${severityClass}">${severityLabel} (${unitSev.toFixed(2)})</span>
                        </div>
                    `;
                    resultsList.appendChild(card);
                });
                
            } catch (err) {
                console.error("NLP Search invocation failed", err);
                resultsList.innerHTML = `<div style="color:var(--danger); padding:2rem; text-align:center;">Search failed: ${err.message}</div>`;
            }
        }

        // Proactive Alerts Logic
        async function loadAlerts() {
            try {
                const response = await fetch('/api/alerts');
                const data = await response.json();
                if (data.status === 'success' && data.alerts && data.alerts.length > 0) {
                    const container = document.getElementById('proactive-alerts-container');
                    container.innerHTML = '';
                    data.alerts.forEach(alert => {
                        const banner = document.createElement('div');
                        banner.className = 'alert-banner';
                        banner.innerHTML = `
                            <svg class="alert-icon" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                            <span class="alert-text">${alert.message}</span>
                        `;
                        container.appendChild(banner);
                    });
                }
            } catch (e) {
                console.error("Error loading alerts:", e);
            }
        }

        // Smart Log Debounce Logic
        function setupSmartLogListener() {
            const input = document.getElementById('smart-log-input');
            if (!input) return;
            
            let debounceTimer;
            input.addEventListener('input', (e) => {
                clearTimeout(debounceTimer);
                const query = e.target.value.trim();
                
                if (query.length < 10) {
                    document.getElementById('smart-log-suggestions').style.display = 'none';
                    return;
                }
                
                debounceTimer = setTimeout(async () => {
                    try {
                        const response = await fetch('/api/smart_log', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: query })
                        });
                        const data = await response.json();
                        
                        const suggestions = data.suggestions; // This is actually the rca_summary dict
                        if (data.status === 'success' && suggestions && Object.keys(suggestions.top_defects || {}).length > 0) {
                            const topDefectsObj = suggestions.top_defects || {};
                            
                            const avgSev = suggestions.avg_severity || 0;
                            const sevLabel = avgSev > 0.6 ? 'High' : (avgSev > 0.3 ? 'Medium' : 'Low');
                            
                            const learnings = suggestions.learnings || [];
                            
                            document.getElementById('smart-log-defect').innerHTML = Object.keys(topDefectsObj).map(k => `<li style="margin-bottom: 0.25rem;">${k} (Count: ${topDefectsObj[k]})</li>`).join('');
                            document.getElementById('smart-log-severity').innerHTML = `<li style="margin-bottom: 0.25rem;">Average: ${avgSev.toFixed(2)} (${sevLabel})</li>`;
                            document.getElementById('smart-log-learning').innerHTML = learnings.length > 0 ? learnings.map(l => `<div style="padding: 1rem; background: white; border-radius: 0.25rem; border: 1px dashed #cbd5e1; font-style: italic;">${l}</div>`).join('') : '<div style="padding: 1rem; background: white; border-radius: 0.25rem; border: 1px dashed #cbd5e1; font-style: italic;">No specific past solution found.</div>';
                            
                            document.getElementById('smart-log-suggestions').style.display = 'block';
                        }
                    } catch (e) {
                        console.error("Smart log fetch failed", e);
                    }
                }, 800);
            });
        }
        
        function submitSmartLog() {
            alert("Complaint Drafted Successfully! (Simulation Mode: Not written to database)");
            document.getElementById('smart-log-input').value = '';
            document.getElementById('smart-log-suggestions').style.display = 'none';
        }