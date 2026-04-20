const state = {
  currentPage: 0,
  dataset: null,
  analysis: null,
  questionDraft: '',
};

const pages = [
  { title: 'Upload data', hint: 'Upload a CSV and describe what the dataset is about.' },
  { title: 'Ask LabMate', hint: 'Ask what you want to know about the actual data.' },
  { title: 'Graph options', hint: 'Review realistic graph suggestions and how to build them.' },
  { title: 'Interpretation', hint: 'Use data-backed observations to explain what the results mean.' },
];

function el(id) { return document.getElementById(id); }

function escapeHtml(text) {
  return (text == null ? '' : String(text)).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[c]));
}

function listHtml(items) {
  return `<ul>${(items || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
}

function setBubble(text) {
  const node = el('guideBubble');
  if (node) node.textContent = text;
}

function setBadge(text, cls = '') {
  const node = el('apiBadge');
  if (!node) return;
  node.className = `badge ${cls}`.trim();
  node.textContent = text;
}

async function checkApi() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    setBadge(data.ok ? 'API ready' : 'API unavailable', data.ok ? 'good' : 'warn');
  } catch (err) {
    setBadge('API unavailable', 'warn');
  }
}

function isLocked(idx) {
  if (idx === 0) return false;
  if (idx === 1) return !state.dataset;
  if (idx === 2) return !state.analysis;
  if (idx === 3) return !state.analysis;
  return false;
}

function renderHeader() {
  el('pageTitle').textContent = pages[state.currentPage].title;
  el('pageHint').textContent = pages[state.currentPage].hint;
}

function renderTabs() {
  const container = el('tabs');
  container.innerHTML = pages.map((page, idx) => {
    const locked = isLocked(idx);
    return `
      <button class="tab ${idx === state.currentPage ? 'active' : ''} ${locked ? 'locked' : ''}" data-tab="${idx}">
        <strong>${idx + 1}. ${escapeHtml(page.title)}</strong>
        <span>${escapeHtml(page.hint)}</span>
      </button>
    `;
  }).join('');

  container.querySelectorAll('[data-tab]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.tab);
      if (!isLocked(idx)) {
        state.currentPage = idx;
        renderAll();
      }
    });
  });
}

async function uploadDataset() {
  const file = el('dataFile')?.files?.[0];
  const description = (el('datasetDescription')?.value || '').trim();

  if (!file) throw new Error('Upload a CSV file.');
  if (!file.name.toLowerCase().endsWith('.csv')) throw new Error('Only CSV files are accepted.');
  if (!description) throw new Error('Describe what the dataset is about.');

  const form = new FormData();
  form.append('file', file);
  form.append('description', description);

  const res = await fetch('/api/upload-data', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Could not upload the dataset.');

  state.dataset = data;
  state.analysis = null;
  state.currentPage = 1;
  setBubble('Nice. Now ask what you want to know. I will use the actual uploaded data to answer, suggest realistic graph options, and explain how to read the result.');
  renderAll();
}

async function askQuestion() {
  const question = (el('questionBox')?.value || '').trim();
  if (!question) throw new Error('Type a question first.');
  if (!state.dataset?.data_id) throw new Error('Upload the CSV first.');

  state.questionDraft = question;
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data_id: state.dataset.data_id, question }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Could not analyze the question.');

  state.analysis = data;
  state.currentPage = 2;
  setBubble('I analyzed the actual data. Start with the graph options tab, then use the interpretation tab to connect the graph back to what the dataset is telling you.');
  renderAll();
}

function renderPreviewTable(rows) {
  if (!rows || !rows.length) return '<p class="small">No preview rows available.</p>';
  const cols = Object.keys(rows[0]);
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${cols.map((col) => `<th>${escapeHtml(col)}</th>`).join('')}</tr>
        </thead>
        <tbody>
          ${rows.map((row) => `<tr>${cols.map((col) => `<td>${escapeHtml(row[col])}</td>`).join('')}</tr>`).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderColumnsTable(columns) {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Column</th>
            <th>Type</th>
            <th>Missing</th>
            <th>Unique</th>
            <th>Sample values</th>
          </tr>
        </thead>
        <tbody>
          ${(columns || []).map((col) => `
            <tr>
              <td>${escapeHtml(col.name)}</td>
              <td>${escapeHtml(col.detected_type)}</td>
              <td>${escapeHtml(col.missing_count)}</td>
              <td>${escapeHtml(col.unique_count)}</td>
              <td>${escapeHtml((col.sample_values || []).join(', '))}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderUploadPage() {
  const page = el('page-0');
  const dataset = state.dataset;

  page.innerHTML = `
    <div class="grid-2">
      <div class="stack">
        <div class="card">
          <div class="pill-title">Required</div>
          <label>CSV file</label>
          <input type="file" id="dataFile" accept=".csv,text/csv" />
          <label style="margin-top:14px;">What is the dataset about?</label>
          <textarea id="datasetDescription" placeholder="Example: Gene expression values for treated vs control samples across three time points.">${escapeHtml(dataset?.description || '')}</textarea>
          <div class="actions">
            <button class="primary" id="uploadBtn">Upload dataset</button>
          </div>
        </div>
        <div class="card">
          <h3>Use LabMate for</h3>
          ${listHtml([
            'Answering what the actual uploaded data says',
            'Choosing realistic graph types for your question',
            'Learning how to build the graph in Python',
            'Interpreting trends, comparisons, spread, relationships, and composition',
          ])}
        </div>
      </div>
      <div class="stack">
        <div class="card">
          <h3>Accepted input</h3>
          ${listHtml([
            'CSV data only',
            'A short description of what the dataset represents',
            'No PDFs or lab manuals',
          ])}
        </div>
        ${dataset ? `
          <div class="card">
            <div class="pill-title">Current dataset</div>
            <h3>${escapeHtml(dataset.filename)}</h3>
            ${listHtml(dataset.overview_bullets || [])}
          </div>
        ` : ''}
      </div>
    </div>
  `;

  el('uploadBtn').addEventListener('click', async () => {
    try {
      await uploadDataset();
    } catch (err) {
      alert(err.message);
    }
  });
}

function renderAskPage() {
  const page = el('page-1');
  const dataset = state.dataset;
  if (!dataset) {
    page.innerHTML = '<div class="card">Upload a dataset first.</div>';
    return;
  }

  page.innerHTML = `
    <div class="stack">
      <div class="grid-2">
        <div class="card">
          <div class="pill-title">Dataset overview</div>
          <h3>${escapeHtml(dataset.filename)}</h3>
          ${listHtml(dataset.overview_bullets || [])}
        </div>
        <div class="card">
          <div class="pill-title">Prompt ideas</div>
          ${listHtml([
            'What graph should I make to compare expression across treatment groups?',
            'Does read depth increase over time?',
            'Are gene_A and gene_B related in this dataset?',
            'How should I show the distribution of cell count?',
            'Which category dominates this dataset?',
            'What does this data mean overall?',
          ])}
        </div>
      </div>
      <div class="card">
        <label>What do you want to know?</label>
        <textarea id="questionBox" placeholder="Ask what the data means, what graph to make, or how to interpret a pattern.">${escapeHtml(state.questionDraft)}</textarea>
        <div class="actions">
          <button class="primary" id="askBtn">Ask LabMate</button>
        </div>
      </div>
      <div class="card">
        <h3>Detected columns</h3>
        ${renderColumnsTable(dataset.columns || [])}
      </div>
    </div>
  `;

  el('askBtn').addEventListener('click', async () => {
    try {
      await askQuestion();
    } catch (err) {
      alert(err.message);
    }
  });
}

function renderGraphPlanPage() {
  const page = el('page-2');
  const analysis = state.analysis;
  if (!analysis) {
    page.innerHTML = '<div class="card">Ask a question first.</div>';
    return;
  }

  page.innerHTML = `
    <div class="stack">
      <div class="card result-block">
        <div class="pill-title">Direct answer</div>
        <h3>What the data says</h3>
        <p>${escapeHtml(analysis.direct_answer)}</p>
        <div class="intent-chip">Intent: ${escapeHtml(analysis.question_intent || 'overview')}</div>
        ${analysis.matched_columns?.length ? `<p class="small" style="margin-top:10px;">Matched columns: ${escapeHtml(analysis.matched_columns.join(', '))}</p>` : ''}
        ${analysis.confidence_note ? `<p class="small" style="margin-top:8px;">${escapeHtml(analysis.confidence_note)}</p>` : ''}
      </div>
      ${(analysis.recommended_graphs || []).map((graph) => `
        <div class="card graph-card">
          <div class="pill-title">${escapeHtml(graph.chart_type)}</div>
          <h3>${escapeHtml(graph.title)}</h3>
          <p>${escapeHtml(graph.why_this_chart)}</p>
          ${graph.good_when ? `<p class="small" style="margin-bottom:14px;"><strong>Best when:</strong> ${escapeHtml(graph.good_when)}</p>` : ''}
          <div class="grid-2">
            <div>
              <h4>How to do it</h4>
              ${listHtml(graph.steps || [])}
            </div>
            <div>
              <h4>How to read it</h4>
              ${listHtml(graph.interpretation_notes || [])}
            </div>
          </div>
          <div style="margin-top:14px;">
            <h4>Python example</h4>
            <pre>${escapeHtml(graph.python_code || '')}</pre>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderInterpretPage() {
  const page = el('page-3');
  const dataset = state.dataset;
  const analysis = state.analysis;
  if (!analysis || !dataset) {
    page.innerHTML = '<div class="card">Ask a question first.</div>';
    return;
  }

  page.innerHTML = `
    <div class="stack">
      <div class="grid-2">
        <div class="card">
          <div class="pill-title">Observed from your data</div>
          <h3>Data-backed observations</h3>
          ${listHtml(analysis.observations || [])}
        </div>
        <div class="card">
          <div class="pill-title">Interpretation help</div>
          <h3>How to explain the result</h3>
          ${listHtml(analysis.interpretation_help || [])}
        </div>
      </div>
      <div class="card">
        <div class="pill-title">Preview rows</div>
        <h3>Sanity check the raw data</h3>
        ${renderPreviewTable(dataset.preview_rows || [])}
      </div>
      <div class="card">
        <div class="pill-title">Ask next</div>
        <h3>Useful follow-ups</h3>
        ${listHtml(analysis.follow_up_questions || [])}
      </div>
    </div>
  `;
}

function renderPages() {
  renderUploadPage();
  renderAskPage();
  renderGraphPlanPage();
  renderInterpretPage();
  document.querySelectorAll('.page').forEach((page, idx) => {
    page.classList.toggle('active', idx === state.currentPage);
  });
}

function renderAll() {
  renderHeader();
  renderTabs();
  renderPages();
}

window.addEventListener('DOMContentLoaded', async () => {
  await checkApi();
  renderAll();
  setBubble('Upload a CSV and describe the dataset. Then ask what you want to know, and I will suggest realistic graph options, show how to make them, and explain what the data actually says.');
});
