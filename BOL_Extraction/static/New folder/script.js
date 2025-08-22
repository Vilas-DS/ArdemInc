document.addEventListener('DOMContentLoaded', () => {
  // --- Core Navigation Logic ---
  const navLinks = document.querySelectorAll('.nav-link');
  const pages = document.querySelectorAll('.page-content');
  const pageTitle = document.getElementById('page-title');

  function showPage(pageId) {
    pages.forEach(page => {
      page.classList.toggle('active', page.id === `${pageId}-page`);
    });
    navLinks.forEach(link => {
      link.classList.toggle('active', link.dataset.page === pageId);
    });
    // Capitalize first letter for title
    pageTitle.textContent = pageId.charAt(0).toUpperCase() + pageId.slice(1);
    
    // Load data for the page if it's the dashboard
    if (pageId === 'dashboard') {
        dashboardModule.load();
    } else if (pageId === 'view-all') {
        // Initial load for history page if it's not already loaded
        if(!document.getElementById('list').hasChildNodes()){
            historyModule.load();
        }
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const pageId = link.dataset.page;
      showPage(pageId);
    });
  });

  // --- Dashboard Module ---
  const dashboardModule = (() => {
    let statusChart = null;
    let trendsChart = null;

    const els = {
        startDate: document.getElementById('dash-start-date'),
        endDate: document.getElementById('dash-end-date'),
        clearBtn: document.getElementById('dash-clear'),
        summary: {
            received: document.getElementById('summary-received'),
            processed: document.getElementById('summary-processed'),
            delivered: document.getElementById('summary-delivered'),
            rejected: document.getElementById('summary-rejected'),
        }
    };

    function buildQuery() {
        const params = new URLSearchParams();
        if (els.startDate.value) params.set('start', els.startDate.value);
        if (els.endDate.value) params.set('end', els.endDate.value);
        return params.toString();
    }

    async function fetchSummary() {
        try {
            const res = await fetch(`/api/dashboard/summary?${buildQuery()}`);
            if (!res.ok) throw new Error('Failed to fetch summary');
            const data = await res.json();
            els.summary.received.textContent = data.received || 0;
            els.summary.processed.textContent = data.processed || 0;
            els.summary.delivered.textContent = data.delivered || 0;
            els.summary.rejected.textContent = data.rejected || 0;
            renderStatusChart(data);
        } catch (error) {
            console.error("Error fetching summary:", error);
        }
    }

    async function fetchTrends() {
        try {
            const res = await fetch(`/api/dashboard/trends?${buildQuery()}`);
            if (!res.ok) throw new Error('Failed to fetch trends');
            const data = await res.json();
            renderTrendsChart(data);
        } catch (error) {
            console.error("Error fetching trends:", error);
        }
    }

    function renderStatusChart(data) {
        const ctx = document.getElementById('status-breakdown-chart').getContext('2d');
        const chartData = {
            labels: ['Received', 'Processed', 'Delivered', 'Rejected'],
            datasets: [{
                data: [data.received, data.processed, data.delivered, data.rejected],
                backgroundColor: ['#3b82f6', '#22c55e', '#f97316', '#ef4444'],
            }]
        };
        if (statusChart) statusChart.destroy();
        statusChart = new Chart(ctx, {
            type: 'doughnut',
            data: chartData,
            options: {
                responsive: true,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    function renderTrendsChart(data) {
        const ctx = document.getElementById('weekly-trends-chart').getContext('2d');
        const chartData = {
            labels: data.labels,
            datasets: [
                {
                    label: 'Processed',
                    data: data.processed,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Received',
                    data: data.received,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        };
        if (trendsChart) trendsChart.destroy();
        trendsChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } },
                plugins: { legend: { position: 'top' } }
            }
        });
    }
    
    function setupEventListeners() {
        els.startDate.addEventListener('change', loadDashboardData);
        els.endDate.addEventListener('change', loadDashboardData);
        els.clearBtn.addEventListener('click', () => {
            els.startDate.value = '';
            els.endDate.value = '';
            loadDashboardData();
        });
    }
    
    function loadDashboardData() {
        fetchSummary();
        fetchTrends();
    }
    
    setupEventListeners();
    
    return { load: loadDashboardData };
  })();

  // --- Uploader Module ---
  const uploaderModule = (() => {
    const ui = {
      uploadButton: document.getElementById('uploadButton'),
      cancelUpload: document.getElementById('cancelUpload'),
      pdfFilesInput: document.getElementById('pdfFiles'),
      dropZone: document.getElementById('drop-zone'),
      fileList: document.getElementById('file-list'),
      errorContainer: document.getElementById('error-container'),
      progressContainer: document.getElementById('progress-container'),
      progressBar: document.getElementById('progress-bar'),
      progressText: document.getElementById('progress-text'),
      processingList: document.getElementById('processing-list'),
    };
    
    let abortUploadCtrl = null;

    ['dragenter','dragover','dragleave','drop'].forEach(ev=>{
      ui.dropZone.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter','dragover'].forEach(()=> ui.dropZone.classList.add('drag-over'));
    ['dragleave','drop'].forEach(()=> ui.dropZone.classList.remove('drag-over'));

    ui.dropZone.addEventListener('drop', e => {
      ui.pdfFilesInput.files = e.dataTransfer.files;
      updateFileList();
    });
    ui.pdfFilesInput.addEventListener('change', updateFileList);

    function updateFileList(){
      ui.fileList.innerHTML = '';
      ui.processingList.innerHTML = '';
      if (ui.pdfFilesInput.files.length>0){
        const list = document.createElement('ul');
        list.className='list-disc list-inside muted';
        Array.from(ui.pdfFilesInput.files).forEach(file=>{
          const li = document.createElement('li');
          li.textContent = `${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
          list.appendChild(li);
          const row = document.createElement('li');
          row.dataset.filename = file.name;
          row.className = 'flex items-center justify-between px-3 py-2 rounded text-sm bg-gray-100';
          row.innerHTML = `<span class="truncate">${file.name}</span><span class="pill pill-wait">Queued</span>`;
          ui.processingList.appendChild(row);
        });
        ui.fileList.appendChild(list);
      }
    }

    function setRowState(name, state){
      const row = Array.from(ui.processingList.children).find(li=> li.dataset.filename===name);
      if (!row) return;
      const pill = row.querySelector('.pill');
      if (!pill) return;
      if (state==='processing'){ pill.className='pill pill-proc'; pill.textContent='Processing'; } 
      else if (state==='done'){ pill.className='pill pill-ok'; pill.textContent='Done'; } 
      else if (state==='error'){ pill.className='pill pill-err'; pill.textContent='Error'; }
    }

    ui.uploadButton.addEventListener('click', handleUpload);
    ui.cancelUpload.addEventListener('click', cancelUpload);

    async function handleUpload(){
      if (ui.pdfFilesInput.files.length===0){ showError('Please select one or more PDF files.'); return; }
      const formData = new FormData();
      Array.from(ui.pdfFilesInput.files).forEach(f=> formData.append('files', f));
      resetUI();
      ui.uploadButton.disabled = true;
      ui.cancelUpload.classList.remove('hidden');
      ui.progressContainer.classList.remove('hidden');
      ui.progressBar.style.width = '5%';
      abortUploadCtrl = new AbortController();
      try{
        const resp = await fetch('/api/extract/', { method:'POST', body: formData, signal: abortUploadCtrl.signal });
        if (!resp.ok) throw new Error(`Server error: ${resp.status} ${resp.statusText}`);
        if (resp.body) {
          await consumeSSE(resp.body.getReader(), onUploadEvent);
        }
      }catch(err){
        if (err.name !== 'AbortError'){ console.error(err); showError('Upload failed. See console.'); }
      }finally{
        ui.cancelUpload.classList.add('hidden');
        ui.uploadButton.disabled = false;
        setTimeout(()=>{ resetUI(); ui.pdfFilesInput.value=''; updateFileList(); }, 1500);
      }
    }
    function cancelUpload(){ if (abortUploadCtrl){ abortUploadCtrl.abort(); } }

    async function consumeSSE(reader, handler){
      const decoder = new TextDecoder();
      let buf = '';
      while (true){
        const {done, value} = await reader.read();
        if (done) break;
        buf += decoder.decode(value, {stream:true});
        const parts = buf.split('\n\n');
        buf = parts.pop() || '';
        for (const chunk of parts){
          const line = chunk.split('\n').find(l=> l.startsWith('data: '));
          if (!line) continue;
          try { handler(JSON.parse(line.slice(6))); } catch(_){}
        }
      }
    }

    function onUploadEvent(data){
      if (data.event === 'start_file') {
        ui.progressText.textContent = `Starting ${data.filename} (${data.fileIndex}/${data.totalFiles})…`;
        setRowState(data.filename, 'processing');
      }
      else if (data.event === 'progress') {
        const pct = Math.max(1, Math.min(99, data.percent ?? 0));
        ui.progressBar.style.width = `${pct}%`;
        ui.progressText.textContent = data.message || `Processing ${data.filename}…`;
      }
      else if (data.event === 'error') {
        showError(`Error processing ${data.filename}: ${data.message||'Unknown error'}`);
        setRowState(data.filename, 'error');
      }
      else if (data.event === 'complete_file') {
        ui.progressBar.style.width = '100%';
        ui.progressBar.classList.add('bg-green-500');
        setRowState(data.filename, 'done');
        // When upload is done, if history page is active, reload it.
        if (document.getElementById('view-all-page').classList.contains('active')) {
            historyModule.load();
        }
      }
      else if (data.event === 'complete_all') {
        ui.progressText.textContent = 'All files processed.';
      }
    }
    
    function showError(msg){
      ui.errorContainer.textContent = msg;
      ui.errorContainer.classList.remove('hidden');
    }
    function resetUI(){
      ui.errorContainer.classList.add('hidden');
      ui.progressContainer.classList.add('hidden');
      ui.progressBar.style.width = '0%';
      ui.progressBar.classList.remove('bg-green-500');
    }
  })();

  // --- History (View All) Module ---
  const historyModule = (() => {
    const PAGE_SIZE = 50;
    let offset = 0;
    const els = {
      form: document.getElementById('filter-form'),
      q: document.getElementById('q'),
      start: document.getElementById('start'),
      end: document.getElementById('end'),
      apply: document.getElementById('apply'),
      reset: document.getElementById('reset'),
      list: document.getElementById('list'),
      empty: document.getElementById('empty'),
      prev: document.getElementById('prev'),
      next: document.getElementById('next'),
      range: document.getElementById('range')
    };

    function buildParams() {
      const p = new URLSearchParams();
      p.set('limit', PAGE_SIZE);
      p.set('offset', offset);
      const q = els.q.value.trim();
      const s = els.start.value;
      const e = els.end.value;
      if (q) p.set('q', q);
      if (s) p.set('start_date', s);
      if (e) p.set('end_date', e);
      return p.toString();
    }

    async function load() {
      const url = `/api/history?` + buildParams();
      const res = await fetch(url);
      const data = await res.json();
      els.list.innerHTML = '';
      if (!Array.isArray(data) || data.length === 0) {
        els.empty.classList.remove('hidden');
        els.range.textContent = '';
        els.next.disabled = true;
        els.next.classList.add('opacity-50', 'cursor-not-allowed');
        return;
      }
      els.empty.classList.add('hidden');
      data.forEach(item => {
        const card = document.createElement('div');
        card.className = 'surface p-4 flex items-center justify-between';
        const metaBits = [
          item.timestamp ? new Date(item.timestamp).toLocaleString() : null,
          (typeof item.pages === 'number') ? `${item.pages} page(s)` : null,
          (typeof item.bills === 'number') ? `${item.bills} bill(s)` : null,
        ].filter(Boolean).join(' • ');
        card.innerHTML = `
          <div class="min-w-0">
            <div class="font-semibold truncate">${item.filename ?? 'Untitled.pdf'}</div>
            <div class="text-xs muted mt-1">${metaBits}</div>
          </div>
          <div class="flex items-center gap-2 flex-shrink-0">
            <a href="/results/${item.id}" target="_blank" class="btn btn-primary text-sm">View</a>
          </div>`;
        els.list.appendChild(card);
      });
      const from = offset + 1;
      const to = offset + data.length;
      els.range.textContent = `Showing ${from}–${to}`;
      els.prev.disabled = offset === 0;
      els.prev.classList.toggle('opacity-50', els.prev.disabled);
      els.prev.classList.toggle('cursor-not-allowed', els.prev.disabled);
      const noMore = data.length < PAGE_SIZE;
      els.next.disabled = noMore;
      els.next.classList.toggle('opacity-50', noMore);
      els.next.classList.toggle('cursor-not-allowed', noMore);
    }

    els.apply.addEventListener('click', (e) => { e.preventDefault(); offset = 0; load(); });
    els.reset.addEventListener('click', () => { els.q.value = ''; els.start.value = ''; els.end.value = ''; offset = 0; load(); });
    els.prev.addEventListener('click', () => { if (offset === 0) return; offset = Math.max(0, offset - PAGE_SIZE); load(); });
    els.next.addEventListener('click', () => { offset += PAGE_SIZE; load(); });
    
    return { load };
  })();

  // --- Initial Load ---
  showPage('dashboard'); // Show dashboard on first load
});
