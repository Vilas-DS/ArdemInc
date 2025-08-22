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
    } else if (pageId === 'uploader') {
        statusModule.checkStatus();
    }
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const pageId = link.dataset.page;
      showPage(pageId);
    });
  });

  // --- System Status Module ---
  const statusModule = (() => {
      const dbIndicator = document.getElementById('db-status-indicator');
      const s3Indicator = document.getElementById('s3-status-indicator');

      const updateIndicator = (el, status, text) => {
          // Base classes are the same for all states
          el.className = 'text-sm font-medium inline-flex items-center px-3 py-1 rounded-full';

          let colorClass = '';
          let icon = '';

          switch (status) {
              case 'connected':
                  colorClass = 'bg-green-100 text-green-800';
                  icon = '<svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>';
                  break;
              case 'checking':
                  colorClass = 'bg-gray-200 text-gray-700';
                  icon = '<svg class="w-4 h-4 mr-2 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';
                  break;
              default: // Covers all error states
                  colorClass = 'bg-red-100 text-red-800';
                  icon = '<svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>';
                  break;
          }
          
          el.classList.add(...colorClass.split(' '));
          el.innerHTML = `${icon} ${text}`;
      };

      async function checkStatus() {
          updateIndicator(dbIndicator, 'checking', 'Checking DB...');
          updateIndicator(s3Indicator, 'checking', 'Checking S3...');

          try {
              const response = await fetch('/api/status');
              if (!response.ok) throw new Error('Status check failed');
              const data = await response.json();

              updateIndicator(dbIndicator, data.db_status, data.db_status === 'connected' ? 'Database Connected' : 'Database Error');

              let s3Text = 'S3 Error';
              if (data.s3_status === 'connected') s3Text = 'S3 Connected';
              else if (data.s3_status === 'not_configured') s3Text = 'S3 Not Configured';
              else if (data.s3_status === 'access_denied') s3Text = 'S3 Access Denied';
              
              updateIndicator(s3Indicator, data.s3_status, s3Text);

          } catch (error) {
              console.error('Failed to fetch system status:', error);
              updateIndicator(dbIndicator, 'error', 'DB Status Unknown');
              updateIndicator(s3Indicator, 'error', 'S3 Status Unknown');
          }
      }
      return { checkStatus };
  })();

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
    let isProcessing = false;

    ['dragenter','dragover','dragleave','drop'].forEach(ev=>{
      ui.dropZone.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); });
    });
    ['dragenter','dragover'].forEach(()=> ui.dropZone.classList.add('drag-over'));
    ['dragleave','drop'].forEach(()=> ui.dropZone.classList.remove('drag-over'));

    ui.dropZone.addEventListener('drop', e => {
      if (isProcessing) return;
      ui.pdfFilesInput.files = e.dataTransfer.files;
      updateFileList();
    });
    ui.pdfFilesInput.addEventListener('change', () => {
        if (isProcessing) return;
        updateFileList();
    });

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
      
      // Don't reset UI, just start the process
      isProcessing = true;
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
        isProcessing = false;
        ui.cancelUpload.classList.add('hidden');
        ui.uploadButton.disabled = false;
        // Reset only after a delay to show completion
        setTimeout(()=>{ 
            ui.progressContainer.classList.add('hidden');
            ui.pdfFilesInput.value='';
            updateFileList();
        }, 2500);
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

  // --- S3 Auto-Import Module ---
  const s3Module = (() => {
    const s3 = {
      toggle: document.getElementById('s3-auto-toggle'),
      scanNow: document.getElementById('s3-scan-now'),
      cancelBtn: document.getElementById('s3-cancel'),
      status: document.getElementById('s3-status'),
      box: document.getElementById('s3-progress'),
      bar: document.getElementById('s3-progress-bar'),
      text: document.getElementById('s3-progress-text'),
      list: document.getElementById('s3-file-list'),
      timerId: null,
      streamES: null,
      running: false,
    };

    s3.scanNow.addEventListener('click', () => startS3Scan(true));

    async function startS3Scan(isManual) {
      if (s3.running) return;
      s3.running = true;
      s3.cancelBtn.classList.remove('hidden');
      s3.box.classList.remove('hidden');
      s3.text.textContent = isManual ? 'Scanning S3 (manual)…' : 'Scanning S3…';
      s3.bar.style.width = '5%';
      s3.list.innerHTML = '';

      try {
        const resp = await fetch('/api/ingest/s3/scan', { method: 'POST' });
        if (!resp.ok) throw new Error(`Scan failed: ${resp.status}`);

        if (s3.streamES) { try { s3.streamES.close(); } catch (_) { } }
        const es = new EventSource('/api/ingest/s3/stream');
        s3.streamES = es;

        let total = 0, done = 0;
        es.onmessage = (ev) => {
          try {
            const data = JSON.parse(ev.data);
            if (data.status === 'processing') {
              total = data.totalFiles || total || 1;
              done = data.currentFile || done;
              s3.text.textContent = `Processing ${data.filename || ''} (${done}/${total})…`;
              s3.bar.style.width = `${Math.round(done / total * 100)}%`;
              appendS3Item(data.filename, 'processing');
            } else if (data.status === 'processed') {
              appendS3Item(data.filename, 'done');
              if (document.getElementById('view-all-page').classList.contains('active')) {
                historyModule.load();
              }
            } else if (data.status === 'error') {
              appendS3Item(data.filename || 'Unknown file', 'error', data.message);
            } else if (data.status === 'complete') {
              s3.text.textContent = 'S3 import complete.';
              s3.bar.style.width = '100%';
              Array.from(s3.list.children).forEach(li => li.className = 'px-3 py-2 rounded text-sm bg-green-100');
              es.close();
              s3.running = false;
              s3.cancelBtn.classList.add('hidden');
            } else if (data.status === 'cancelled') {
              s3.text.textContent = 'S3 import cancelled.';
              s3.bar.style.width = '0%';
              es.close();
              s3.running = false;
              s3.cancelBtn.classList.add('hidden');
            }
          } catch (_) { }
        };

        es.onerror = () => {
          s3.running = false;
          s3.cancelBtn.classList.add('hidden');
        };
      } catch (e) {
        s3.text.textContent = e.message || 'Scan error';
        s3.running = false;
        s3.cancelBtn.classList.add('hidden');
      }
    }

    function appendS3Item(filename, state, msg) {
      if (!filename) return;
      let li = Array.from(s3.list.children).find(el => el.dataset.filename === filename);
      if (!li) {
          li = document.createElement('li');
          li.dataset.filename = filename;
          s3.list.appendChild(li);
      }
      
      li.textContent = filename + (msg ? ` — ${msg}` : '');
      li.className = 'px-3 py-2 rounded text-sm ' + (
        state === 'error' ? 'bg-red-100' :
          state === 'processing' ? 'bg-blue-100' :
            'bg-green-100'
      );
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
