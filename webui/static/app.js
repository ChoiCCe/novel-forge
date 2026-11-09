'use strict';

const S = {
  bookId: null,
  book: null,
  summary: null,
  chapters: [],
  offset: 0,
  limit: 100,
  voices: [],
  stages: [],
  cast: [],
  changes: [],
  tasks: [],
  temperaments: [],
  settings: null,
  segments: [],
  pollTimer: null,
};

const STAGE_LABEL = { script: '剧本', coverage: '覆盖率', tts: '合成', export: '导出' };
const STATUS_LABEL = { queued: '排队中', running: '执行中', paused: '已暂停', cancelled: '已取消', done: '完成', failed: '有失败' };
const TEMPERAMENT_FALLBACK = ['沉稳', '急躁', '阴柔', '豪爽', '阴沉', '中性'];

// ------------------------------------------------------------------ 基础
async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  let data;
  try { data = await res.json(); } catch (e) { data = { ok: false, error: '响应解析失败' }; }
  if (!res.ok || data.ok === false) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

function esc(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmtBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes, i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return value.toFixed(value >= 10 ? 1 : 2) + ' ' + units[i];
}

function fmtDuration(ms) {
  if (!ms) return '—';
  const total = Math.round(ms / 1000);
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(s).padStart(2, '0');
}

function toast(message, bad = false) {
  const node = document.createElement('div');
  node.className = 'toast' + (bad ? ' bad' : '');
  node.textContent = message;
  document.body.appendChild(node);
  setTimeout(() => node.remove(), bad ? 5000 : 2600);
}

// ------------------------------------------------------------------ 环境
async function loadHealth() {
  const node = document.getElementById('health');
  try {
    const data = await api('/api/health');
    const dot = '<span class="dot ' + (data.edge ? 'ok' : 'bad') + '"></span>';
    node.innerHTML = dot + 'edge ' + (data.edge ? '就绪' : '不可用') +
      ' ｜ ffmpeg ' + (data.ffmpeg && !data.ffmpeg.startsWith('不可用') ? '就绪' : esc(data.ffmpeg || '未知'));
  } catch (err) {
    node.innerHTML = '<span class="dot bad"></span>' + esc(err.message);
  }
}

// ------------------------------------------------------------------ 书库
async function loadBooks() {
  try {
    const data = await api('/api/books');
    const box = document.getElementById('books');
    if (!data.books.length) {
      box.innerHTML = '<p class="hint">还没有书，先把 txt 拖进上面的区域。</p>';
      return;
    }
    box.innerHTML = data.books.map(book => `
      <div class="book ${book.id === S.bookId ? 'active' : ''}" data-id="${esc(book.id)}">
        <div class="name">《${esc(book.title)}》</div>
        <div class="meta">${book.chapter_count} 章 ｜ ${(book.total_chars / 10000).toFixed(1)} 万字 ｜ ${esc(book.encoding || '')}</div>
        <div class="meta">${esc(book.id)}</div>
      </div>`).join('');
    box.querySelectorAll('.book').forEach(node => {
      node.addEventListener('click', () => selectBook(node.dataset.id));
    });
    const select = document.getElementById('copySourceSelect');
    if (select) {
      const others = data.books.filter(book => book.id !== S.bookId);
      select.innerHTML = '<option value="">选择来源书…</option>' + others.map(book =>
        `<option value="${esc(book.id)}">《${esc(book.title)}》</option>`).join('');
    }
  } catch (err) { toast(err.message, true); }
}

let uploading = false;
async function uploadFile(file) {
  if (!file) return;
  if (uploading) return;
  uploading = true;
  try {
    toast('正在导入《' + file.name + '》…');
    const res = await fetch('/api/books/upload?filename=' + encodeURIComponent(file.name),
      { method: 'POST', body: file });
    const data = await res.json();
    if (!res.ok || data.ok === false) throw new Error(data.error || '导入失败');
    toast('导入完成：' + data.book.chapter_count + ' 章');
    await loadBooks();
    selectBook(data.book_id);
  } catch (err) { toast(err.message, true); }
  finally { uploading = false; }
}

async function importPath() {
  const path = document.getElementById('pathInput').value.trim();
  if (!path) { toast('先填一个路径', true); return; }
  try {
    const data = await api('/api/books/import', { method: 'POST', body: JSON.stringify({ path }) });
    toast('导入完成：' + data.book.chapter_count + ' 章');
    await loadBooks();
    selectBook(data.book_id);
  } catch (err) { toast(err.message, true); }
}

function bindDropzone() {
  const zone = document.getElementById('dropzone');
  const input = document.getElementById('fileInput');
  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => uploadFile(input.files[0]));
  ['dragover', 'dragenter'].forEach(name => zone.addEventListener(name, event => {
    event.preventDefault(); zone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => {
    event.preventDefault(); zone.classList.remove('dragging');
  }));
  zone.addEventListener('drop', event => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
}

// ------------------------------------------------------------------ 章节
async function selectBook(bookId) {
  S.bookId = bookId;
  S.offset = 0;
  try {
    const data = await api('/api/books/' + encodeURIComponent(bookId));
    S.book = data.book;
    S.summary = data.summary;
  } catch (err) { toast(err.message, true); return; }
  document.getElementById('chapterSection').hidden = false;
  document.getElementById('castSection').hidden = false;
  document.getElementById('settingsSection').hidden = false;
  document.getElementById('segmentSection').hidden = false;
  document.getElementById('taskSection').hidden = false;
  document.getElementById('outputSection').hidden = false;
  document.getElementById('bookTitle').textContent = S.book ? '《' + S.book.title + '》章节' : '章节';
  await loadBooks();
  await Promise.all([loadChapters(), loadCast(), loadSettings(), loadTasks(), loadOutputs()]);
}

async function loadChapters() {
  if (!S.bookId) return;
  try {
    const data = await api(`/api/books/${encodeURIComponent(S.bookId)}/chapters?offset=${S.offset}&limit=${S.limit}`);
    S.chapters = data.chapters;
    S.summary = data.summary;
    renderChapters();
    renderSummary();
  } catch (err) { toast(err.message, true); }
}

function renderSummary() {
  const s = S.summary || {};
  document.getElementById('bookStats').innerHTML = `
    <span>章节 <b>${s.chapter_count || 0}</b></span>
    <span>总字数 <b>${((s.total_chars || 0) / 10000).toFixed(2)} 万</b></span>
    <span>已选 <b>${s.selected_count || 0}</b> 章 / <b>${((s.selected_chars || 0) / 10000).toFixed(2)} 万</b> 字</span>
    <span>预估时长 <b>${fmtDuration(s.est_duration_ms || 0)}</b></span>`;
  const total = s.chapter_count || 0;
  document.getElementById('chapterRange').textContent =
    total ? `第 ${S.offset + 1}–${Math.min(S.offset + S.limit, total)} 章 / 共 ${total} 章` : '';
  document.getElementById('prevPage').disabled = S.offset <= 0;
  document.getElementById('nextPage').disabled = S.offset + S.limit >= total;
}

function renderChapters() {
  const body = document.getElementById('chapterBody');
  body.innerHTML = S.chapters.map(row => `
    <tr data-id="${row.id}">
      <td><input type="checkbox" class="sel" ${row.selected ? 'checked' : ''}></td>
      <td>${row.idx}${row.label ? ' <span class="hint">' + esc(row.label) + '</span>' : ''}</td>
      <td><input type="text" class="title" value="${esc(row.title)}"></td>
      <td class="num">${row.char_count}</td>
      <td class="num">${fmtDuration(row.est_duration_ms)}</td>
    </tr>`).join('');
  body.querySelectorAll('input.sel').forEach(node => {
    node.addEventListener('change', () => saveChapter(node.closest('tr')));
  });
  body.querySelectorAll('input.title').forEach(node => {
    node.addEventListener('change', () => saveChapter(node.closest('tr')));
  });
}

async function saveChapter(tr) {
  const id = tr.dataset.id;
  const title = tr.querySelector('input.title').value;
  const selected = tr.querySelector('input.sel').checked ? 1 : 0;
  try {
    const data = await api('/api/chapters/' + id, {
      method: 'PATCH', body: JSON.stringify({ title, selected }),
    });
    S.summary = data.summary;
    renderSummary();
  } catch (err) { toast(err.message, true); }
}

async function setSelection(mode) {
  if (!S.bookId) return;
  try {
    await api(`/api/books/${encodeURIComponent(S.bookId)}/selection`, {
      method: 'POST', body: JSON.stringify({ mode }),
    });
    await loadChapters();
  } catch (err) { toast(err.message, true); }
}

function changePage(delta) {
  const total = (S.summary && S.summary.chapter_count) || 0;
  const next = S.offset + delta * S.limit;
  if (next < 0 || next >= total) return;
  S.offset = next;
  loadChapters();
}

// ------------------------------------------------------------------ 角色与音色
async function loadCast() {
  if (!S.bookId) return;
  try {
    const data = await api(`/api/books/${encodeURIComponent(S.bookId)}/cast`);
    S.voices = data.voices;
    S.stages = data.stages;
    S.cast = data.cast;
    S.changes = data.changes;
    S.narrator = data.narrator;
    S.collisions = data.collisions || { by_voice: {}, warnings: [] };
    S.drift = data.drift || [];
    S.bookOptions = data.book_options || {};
    S.tts = data.tts || {};
    const realAge = document.getElementById('realAgeInput');
    if (realAge) realAge.checked = !!S.bookOptions.real_age;
    const conc = document.getElementById('ttsConcurrencyInput');
    const interval = document.getElementById('ttsIntervalInput');
    if (conc) conc.value = S.tts.concurrency;
    if (interval) interval.value = S.tts.min_interval_s;
    const ttsHint = document.getElementById('ttsHint');
    if (ttsHint && S.tts.concurrency) {
      ttsHint.textContent = `并发 ${S.tts.concurrency}（上限 ${S.tts.max_concurrency}）/ 间隔 ${S.tts.min_interval_s}s`;
    }
    renderCast(S.narrator);
  } catch (err) { toast(err.message, true); }
}

function effText(obj) {
  if (!obj) return '—';
  return `${esc(obj.rate)} / ${esc(obj.pitch)} / ${esc(obj.volume)}`;
}

const SOURCE_LABEL = {
  manual: '手动', locked: '锁定', migrated: '迁移', auto: '自动',
  config: '配置', copy: '复制', fallback: '兜底', inferred: '推断',
};

function sourceBadge(row) {
  const key = row.voice_source || 'config';
  return `<span class="badge">${esc(SOURCE_LABEL[key] || key)}</span>`;
}

function collisionFor(voice) {
  const byVoice = (S.collisions && S.collisions.by_voice) || {};
  return byVoice[voice] || null;
}

function ageHistoryText(row) {
  const items = row.age_history || [];
  if (!items.length) return '—';
  return items.map(item =>
    `${esc(item.stage)}${item.first_seen_chapter ? '（第' + item.first_seen_chapter + '章起）' : ''}`).join(' → ');
}

function voiceOptions(current) {
  const list = S.voices.slice();
  if (current && !list.includes(current)) list.unshift(current);
  return list;
}

function renderCast(narrator) {
  const body = document.getElementById('castBody');
  const temperaments = S.temperaments.length ? S.temperaments : TEMPERAMENT_FALLBACK;
  const rows = [];
  if (narrator) {
    const voices = voiceOptions(narrator.edge_voice);
    rows.push(`<tr class="narrator" data-narrator="1">
      <td>旁白${sourceBadge(narrator)}</td><td>女</td><td>${esc(narrator.age_stage)}</td><td>中性</td>
      <td><select class="voice">${voices.map(voice =>
        `<option ${voice === narrator.edge_voice ? 'selected' : ''}>${esc(voice)}</option>`).join('')}</select></td>
      <td><input type="text" class="rate" value="${esc(narrator.rate || '+0%')}"></td>
      <td><input type="text" class="pitch" value="${esc(narrator.pitch || '+0Hz')}"></td>
      <td><input type="text" class="volume" value="${esc(narrator.volume || '+0%')}"></td>
      <td class="eff">${effText(narrator.effective)}</td>
      <td class="hint">全章旁白，跨章锁死${narrator.legacy_edge_voice
        ? '<br>历史声线 ' + esc(narrator.legacy_edge_voice) + ' 已失效，已平替'
        : ''}</td>
      <td></td>
      <td><button class="ghost" data-act="save-narrator">保存</button>
        <button class="ghost" data-act="preview-narrator">试听</button>
        <label class="check"><input type="checkbox" class="global-scope">全局</label></td>
    </tr>`);
  }
  if (!S.cast.length) {
    rows.push('<tr><td colspan="12" class="hint">还没有角色，先跑一章（或点“开始合成”）生成角色表。</td></tr>');
  }
  for (const row of S.cast) {
    const voices = voiceOptions(row.edge_voice);
    const coll = collisionFor(row.edge_voice);
    const collBadge = coll && coll.level !== 'ok'
      ? `<span class="badge ${coll.level === 'error' ? 'err' : 'warn'}" title="${esc((coll.members || []).join('、'))}">撞声 ${coll.score}</span>`
      : '';
    const genderBadge = (row.gender === '未知' || row.gender_confirmed === false)
      ? `<span class="badge warn" title="${esc(row.gender_evidence || '无性别证据，待确认')}">待确认</span>`
      : '';
    const lockBadge = row.locked ? `<span class="badge info">锁定</span>` : '';
    const transition = row.transition
      ? `<br>渐变 ${esc(row.transition.from_stage)}→${esc(row.transition.to_stage)}（第${row.transition.start_chapter}章起 ${row.transition.chapters}章）`
      : '';
    const hold = row.age_hold_reason ? `<br>未升阶：${esc(row.age_hold_reason)}` : '';
    const legacy = row.legacy_edge_voice
      ? `<br>历史声线 ${esc(row.legacy_edge_voice)} 已失效：${esc(row.voice_repair || '已平替')}`
      : '';
    const isException = String(row.binding_id || '').startsWith('book_character_stage:');
    const exceptionBtn = isException
      ? '<button class="ghost" data-act="rm-exception">取消本例外</button>'
      : '<button class="ghost" data-act="exception">本书例外</button>';
    rows.push(`<tr data-key="${esc(row.key)}">
      <td>${esc(row.name)}<br>${sourceBadge(row)}${lockBadge}${genderBadge}${collBadge}</td>
      <td>${esc(row.gender || '—')}</td>
      <td><select class="stage">${S.stages.map(stage =>
        `<option ${stage === row.age_stage ? 'selected' : ''}>${esc(stage)}</option>`).join('')}</select></td>
      <td><select class="temperament">${temperaments.map(name =>
        `<option ${name === row.temperament ? 'selected' : ''}>${esc(name)}</option>`).join('')}</select></td>
      <td><select class="voice">${voices.map(voice =>
        `<option ${voice === row.edge_voice ? 'selected' : ''}>${esc(voice)}</option>`).join('')}</select></td>
      <td><input type="text" class="rate" value="${esc(row.rate || '+0%')}"></td>
      <td><input type="text" class="pitch" value="${esc(row.pitch || '+0Hz')}"></td>
      <td><input type="text" class="volume" value="${esc(row.volume || '+0%')}"></td>
      <td class="eff">${effText(row.effective)}</td>
      <td class="hint">${esc(row.age_note || '—')}<br>
        性别置信 ${row.gender_confidence == null ? '—' : row.gender_confidence.toFixed(2)}
        ｜ 年龄置信 ${row.age_confidence == null ? '—' : row.age_confidence.toFixed(2)}
        ｜ 首见第 ${row.first_seen_chapter || '?'} 章<br>
        年龄史：${ageHistoryText(row)}${transition}${hold}${legacy}
      </td>
      <td><select class="voice-b"><option value="">—</option>${S.voices.map(voice =>
        `<option>${esc(voice)}</option>`).join('')}</select></td>
      <td><button class="ghost" data-act="save">保存</button>
        <button class="ghost" data-act="preview">试听</button>
        <button class="ghost" data-act="ab">A/B</button>
        ${exceptionBtn}</td>
    </tr>`);
  }
  body.innerHTML = rows.join('');
  const narratorRow = body.querySelector('tr[data-narrator]');
  if (narratorRow) {
    narratorRow.querySelector('[data-act="save-narrator"]').addEventListener('click',
      () => saveNarrator(narratorRow));
    narratorRow.querySelector('[data-act="preview-narrator"]').addEventListener('click',
      () => previewNarrator(narratorRow));
  }
  body.querySelectorAll('tr[data-key]').forEach(tr => {
    tr.querySelector('[data-act="save"]').addEventListener('click', () => saveCast(tr));
    tr.querySelector('[data-act="preview"]').addEventListener('click', () => previewCast(tr));
    tr.querySelector('[data-act="ab"]').addEventListener('click', () => abCast(tr));
    const ex = tr.querySelector('[data-act="exception"]');
    if (ex) ex.addEventListener('click', () => saveBookException(tr));
    const rm = tr.querySelector('[data-act="rm-exception"]');
    if (rm) rm.addEventListener('click', () => removeBookException(tr));
  });
  renderCollisionHint();
  renderChanges();
}

function renderCollisionHint() {
  const node = document.getElementById('collisionHint');
  if (!node) return;
  const warnings = (S.collisions && S.collisions.warnings) || [];
  const drift = S.drift || [];
  const parts = [];
  if (warnings.length) {
    parts.push(`${warnings.filter(w => w.level === 'error').length} 条阻断 / ` +
      `${warnings.filter(w => w.level === 'warn').length} 条警告`);
  }
  if (drift.length) parts.push(`${drift.length} 条配置漂移`);
  node.textContent = parts.length ? '一致性：' + parts.join('，') : '一致性：未发现撞声/漂移';
  node.className = 'hint' + (warnings.some(w => w.level === 'error') ? ' err' : '');
}

function renderChanges() {
  const box = document.getElementById('changeList');
  if (!S.changes.length) {
    box.innerHTML = '<span class="hint">暂无换音色记录。跨阶段或手动调整声线时会记录在这里。</span>';
    return;
  }
  box.innerHTML = S.changes.map(item => `
    <div>第 ${esc(item.chapter_idx)} 章 · ${esc(item.name)}：${esc(item.from_stage)} ${esc(item.from_voice)} → ${esc(item.to_stage)} ${esc(item.to_voice)}
      <span class="hint">（${esc(item.evidence || '')}，${esc(item.updated_by || '')}）</span></div>`).join('');
}

async function saveCast(tr) {
  const key = tr.dataset.key;
  const row = S.cast.find(item => item.key === key);
  const stage = tr.querySelector('.stage').value;
  const voice = tr.querySelector('.voice').value;
  const temperament = tr.querySelector('.temperament').value;
  const rate = tr.querySelector('.rate').value.trim() || '+0%';
  const pitch = tr.querySelector('.pitch').value.trim() || '+0Hz';
  const volume = tr.querySelector('.volume').value.trim() || '+0%';
  if (row && stage !== row.age_stage) {
    const ok = confirm(`确认把「${row.name}」从「${row.age_stage}」推进到「${stage}」？\n` +
      `会锁定新的声线并写入换音色历史。`);
    if (!ok) return;
  }
  if (row && temperament !== row.temperament) {
    toast('性格已改为「' + temperament + '」，基础韵律将按该性格 + 年龄段重算');
  }
  try {
    const data = await api(`/api/cast/${encodeURIComponent(S.bookId)}/${encodeURIComponent(key)}`, {
      method: 'PATCH',
      body: JSON.stringify({ age_stage: stage, edge_voice: voice, rate, pitch, volume, temperament }),
    });
    S.cast = data.cast;
    S.changes = data.changes;
    S.collisions = data.collisions || S.collisions;
    S.drift = data.drift || S.drift;
    renderCast(S.narrator);
    toast('已保存');
  } catch (err) { toast(err.message, true); }
}

function castPayload(tr, voice) {
  return {
    book_id: S.bookId, key: tr.dataset.key,
    edge_voice: voice,
    temperament: tr.querySelector('.temperament').value,
    rate: tr.querySelector('.rate').value.trim(),
    pitch: tr.querySelector('.pitch').value.trim(),
    volume: tr.querySelector('.volume').value.trim(),
  };
}

async function synthCast(tr, voice) {
  return api('/api/preview', { method: 'POST', body: JSON.stringify(castPayload(tr, voice)) });
}

async function playPreview(data, label) {
  const audio = document.getElementById('previewAudio');
  const hint = document.getElementById('previewHint');
  audio.src = data.url + '&t=' + Date.now();
  await audio.play().catch(() => {});
  hint.textContent = `${label ? label + ' ｜ ' : ''}${data.voice} ｜ ` +
    `${data.rate} / ${data.pitch} / ${data.volume} ｜ ` +
    `${data.duration_ms / 1000}s ｜ 「${data.text}」`;
  return data;
}

async function previewCast(tr) {
  const hint = document.getElementById('previewHint');
  hint.textContent = '合成试听中…';
  try {
    const data = await synthCast(tr, tr.querySelector('.voice').value);
    await playPreview(data);
  } catch (err) { hint.textContent = ''; toast(err.message, true); }
}

async function abCast(tr) {
  const current = tr.querySelector('.voice').value;
  const alt = tr.querySelector('.voice-b').value;
  if (!alt) { toast('先在“A/B 备选”里选第二把声线', true); return; }
  const hint = document.getElementById('previewHint');
  hint.textContent = `A/B 合成中：${current} vs ${alt}…`;
  try {
    const first = await synthCast(tr, current);
    const second = await synthCast(tr, alt);
    await playPreview(first, 'A');
    await new Promise(resolve => setTimeout(resolve, 700));
    await playPreview(second, 'B');
    hint.textContent = `A ${current} ${(first.duration_ms / 1000).toFixed(2)}s ｜ ` +
      `B ${alt} ${(second.duration_ms / 1000).toFixed(2)}s ｜ 差值 ` +
      `${((second.duration_ms - first.duration_ms) / 1000).toFixed(2)}s`;
  } catch (err) { hint.textContent = ''; toast(err.message, true); }
}

function rowOf(tr) {
  return S.cast.find(item => item.key === tr.dataset.key) || {};
}

async function saveBookException(tr) {
  const voice = tr.querySelector('.voice').value;
  if (!confirm('创建“本书例外”会覆盖该作者的全局声线，仅对本书生效。继续？')) return;
  try {
    const data = await api('/api/cast/binding', {
      method: 'POST',
      body: JSON.stringify({
        book_id: S.bookId, key: tr.dataset.key, edge_voice: voice,
        scope: 'book_character_stage', force: true,
      }),
    });
    S.cast = data.cast;
    S.collisions = data.collisions || S.collisions;
    renderCast(S.narrator);
    toast(data.binding && data.binding.id ? '已创建本书例外' : '已保存');
  } catch (err) { toast(err.message, true); }
}

async function removeBookException(tr) {
  const row = rowOf(tr);
  const bindingId = row.binding_id || '';
  if (!bindingId.startsWith('book_character_stage:')) { toast('当前不是本书例外', true); return; }
  if (!confirm('删除本书例外，恢复全局声线？')) return;
  try {
    const data = await api('/api/cast/binding', {
      method: 'POST',
      body: JSON.stringify({
        book_id: S.bookId, key: tr.dataset.key, edge_voice: row.edge_voice,
        scope: 'book_character_stage', binding_id: bindingId, remove: true,
      }),
    });
    S.cast = data.cast;
    S.collisions = data.collisions || S.collisions;
    renderCast(S.narrator);
    toast('已删除本书例外');
  } catch (err) { toast(err.message, true); }
}

async function saveNarrator(tr) {
  const scope = tr.querySelector('.global-scope').checked ? 'global' : 'book';
  try {
    const data = await api('/api/narrator', {
      method: 'POST',
      body: JSON.stringify({
        book_id: S.bookId, scope,
        edge_voice: tr.querySelector('.voice').value,
        rate: tr.querySelector('.rate').value.trim(),
        pitch: tr.querySelector('.pitch').value.trim(),
        volume: tr.querySelector('.volume').value.trim(),
      }),
    });
    S.narrator = data.narrator;
    renderCast(S.narrator);
    toast(scope === 'global' ? '旁白已写入全局' : '旁白已保存到本书');
  } catch (err) { toast(err.message, true); }
}

async function previewNarrator(tr) {
  const hint = document.getElementById('previewHint');
  hint.textContent = '旁白试听合成中…';
  try {
    const data = await api('/api/preview/narrator', {
      method: 'POST',
      body: JSON.stringify({
        book_id: S.bookId,
        edge_voice: tr.querySelector('.voice').value,
        rate: tr.querySelector('.rate').value.trim(),
        pitch: tr.querySelector('.pitch').value.trim(),
        volume: tr.querySelector('.volume').value.trim(),
        text: '这一章的开头，风从山口吹下来，带着一点凉意。',
      }),
    });
    await playPreview(data, '旁白');
  } catch (err) { hint.textContent = ''; toast(err.message, true); }
}

async function toggleRealAge(input) {
  if (!S.bookId) return;
  try {
    const data = await api('/api/book-options', {
      method: 'POST',
      body: JSON.stringify({ book_id: S.bookId, key: 'real_age', value: input.checked }),
    });
    S.bookOptions = data.book_options;
    toast(input.checked ? '本书已改为按真实年龄换声' : '本书恢复缓冲线（18 岁仍少年）');
  } catch (err) { input.checked = !input.checked; toast(err.message, true); }
}

async function saveBookTts() {
  if (!S.bookId) return;
  const conc = document.getElementById('ttsConcurrencyInput');
  const interval = document.getElementById('ttsIntervalInput');
  try {
    const first = await api('/api/book-options', {
      method: 'POST',
      body: JSON.stringify({ book_id: S.bookId, key: 'tts_concurrency', value: conc.value }),
    });
    const second = await api('/api/book-options', {
      method: 'POST',
      body: JSON.stringify({ book_id: S.bookId, key: 'tts_min_interval_s', value: interval.value }),
    });
    S.bookOptions = second.book_options;
    S.tts = second.tts || first.tts || S.tts;
    const hint = document.getElementById('ttsHint');
    if (hint) hint.textContent = `本书：并发 ${S.tts.concurrency} / 间隔 ${S.tts.min_interval_s}s（${S.tts.source === 'book' ? '单书覆盖' : '全局默认'}）`;
    toast('本书 TTS 调度已保存');
  } catch (err) { toast(err.message, true); }
}

async function copyCastConfig() {
  const select = document.getElementById('copySourceSelect');
  const source = select.value;
  if (!source) { toast('先选一本来源书', true); return; }
  if (!confirm('把来源书中同名角色的声线复制为本书的显式例外绑定？')) return;
  try {
    const data = await api('/api/cast/import', {
      method: 'POST',
      body: JSON.stringify({ book_id: S.bookId, source_book_id: source }),
    });
    S.cast = data.cast;
    S.collisions = data.collisions || S.collisions;
    renderCast(S.narrator);
    toast(`已复制 ${data.created.length} 个，跳过 ${data.skipped.length} 个`);
    if (data.conflicts.length) {
      alert('跨书冲突（已建单书例外）：\n' + data.conflicts.map(item =>
        `${item.name}：本书 ${item.target_voice} / 来源 ${item.source_voice}`).join('\n'));
    }
  } catch (err) { toast(err.message, true); }
}

// ------------------------------------------------------------------ 导演层设置
async function loadDirectorEstimate() {
  const node = document.getElementById('directorEstimate');
  if (!S.bookId) { node.textContent = '预计送审 —（先选择书）'; return; }
  try {
    S.directorEstimate = await api(`/api/books/${encodeURIComponent(S.bookId)}/director-estimate`);
    renderDirectorEstimate();
  } catch (err) { node.textContent = '预计送审 —'; }
}

function renderDirectorEstimate() {
  const node = document.getElementById('directorEstimate');
  const d = S.directorEstimate;
  if (!node || !d) return;
  const on = document.getElementById('directorInput').checked;
  if (!d.total) {
    node.textContent = `预计送审 0 段（${d.missing_chapters || 0} 章尚未生成剧本）`;
    return;
  }
  const missing = d.missing_chapters ? `；另有 ${d.missing_chapters} 章尚未生成剧本` : '';
  node.textContent = on
    ? `预计送审 ${d.sent}/${d.total} 段（默认 ${d.default_count} 段${missing}）`
    : `导演已关闭（开启后预计送审 ${d.sent}/${d.total} 段${missing}）`;
}

async function loadSettings() {
  try {
    const data = await api('/api/settings');
    S.settings = data;
    S.temperaments = data.temperaments || [];
    const director = document.getElementById('directorInput');
    director.checked = !!(data.director && data.director.enabled);
    director.onchange = renderDirectorEstimate;
    const template = data.pause_template || {};
    for (const key of Object.keys(template)) {
      const input = document.getElementById('pause_' + key);
      if (input) input.value = template[key];
    }
    renderCast(S.narrator);
    loadDirectorEstimate();
  } catch (err) { toast(err.message, true); }
}

async function saveSettings() {
  const template = {};
  for (const key of ['chapter_start_ms', 'narration_to_dialogue_ms',
    'dialogue_to_narration_ms', 'segment_gap_ms', 'chapter_end_ms']) {
    const input = document.getElementById('pause_' + key);
    template[key] = input ? Number(input.value || 0) : 0;
  }
  try {
    const data = await api('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify({
        director_enabled: document.getElementById('directorInput').checked,
        pause_template: template,
      }),
    });
    S.settings = data;
    toast('设置已保存（下一次任务生效）');
  } catch (err) { toast(err.message, true); }
}

// ------------------------------------------------------------------ 单段试听
async function loadSegments() {
  if (!S.bookId) return;
  const chapter = Number(document.getElementById('segChapter').value || 1);
  const hint = document.getElementById('segHint');
  hint.textContent = '加载中…';
  try {
    const data = await api(`/api/books/${encodeURIComponent(S.bookId)}/chapters/${chapter}/segments`);
    S.segments = data.segments;
    S.maxSegmentFailures = Number(data.max_failures || 3);
    renderSegments();
    hint.textContent = data.segments.length
      ? `${data.segments.length} 段（第 ${chapter} 章）`
      : '该章还没有剧本分段，先跑一次本章。';
  } catch (err) { hint.textContent = ''; toast(err.message, true); }
}

function segmentStatus(row) {
  const max = S.maxSegmentFailures || 3;
  const count = Number(row.fail_count || 0);
  if (!count) return '<span class="hint">—</span>';
  const tip = `${esc(row.last_error_type || '未知错误')} · ${esc(row.error || '')} · ${esc(row.last_error_at || '')}`;
  const level = count >= max ? 'badge-fail' : 'badge-warn';
  const label = count >= max ? `失败 ${count} 次（已停止自动重试）` : `失败 ${count} 次`;
  return `<span class="badge ${level}" title="${tip}">${label}</span>`;
}

function renderSegments() {
  const body = document.getElementById('segmentBody');
  if (!S.segments.length) {
    body.innerHTML = '<tr><td colspan="7" class="hint">没有可试听的分段。</td></tr>';
    return;
  }
  body.innerHTML = S.segments.map(row => `
    <tr data-job="${row.job_index}" data-chapter="${row.chapter_idx}">
      <td>${row.job_index}</td>
      <td>${row.type === 'dialogue' ? '对白' : '旁白'}</td>
      <td>${esc(row.speaker)}</td>
      <td>${esc(row.emotion)}</td>
      <td>${segmentStatus(row)}</td>
      <td class="seg-text">${esc(row.text)}</td>
      <td>
        <button class="ghost" data-act="seg-preview">试听</button>
        ${Number(row.fail_count || 0) ? '<button class="ghost" data-act="seg-retry">重试</button>' : ''}
      </td>
    </tr>`).join('');
  body.querySelectorAll('[data-act="seg-preview"]').forEach(button => {
    button.addEventListener('click', () => previewSegment(button.closest('tr')));
  });
  body.querySelectorAll('[data-act="seg-retry"]').forEach(button => {
    button.addEventListener('click', () => retrySegment(button.closest('tr')));
  });
}

async function retrySegment(tr) {
  try {
    const data = await api(`/api/books/${encodeURIComponent(S.bookId)}/chapters/` +
      `${tr.dataset.chapter}/segments/${tr.dataset.job}/retry`, {
      method: 'POST',
      body: JSON.stringify({ force: false }),
    });
    toast(`已重置该分段失败计数，任务 #${data.task_id} 已入队`);
    loadSegments();
    loadTasks();
  } catch (err) { toast(err.message, true); }
}

async function previewSegment(tr) {
  const hint = document.getElementById('segmentPreviewHint');
  hint.textContent = '合成中…';
  try {
    const data = await api('/api/preview/segment', {
      method: 'POST',
      body: JSON.stringify({
        book_id: S.bookId,
        chapter_idx: Number(tr.dataset.chapter),
        job_index: Number(tr.dataset.job),
      }),
    });
    const audio = document.getElementById('segmentAudio');
    audio.src = data.url + '&t=' + Date.now();
    await audio.play().catch(() => {});
    const direction = data.direction || {};
    hint.textContent = `${data.voice} ｜ ${data.rate} / ${data.pitch} / ${data.volume} ｜ ` +
      `pace ${direction.pace || 'normal'} ｜ ${data.duration_ms / 1000}s`;
  } catch (err) { hint.textContent = ''; toast(err.message, true); }
}

// ------------------------------------------------------------------ 任务
async function createTask() {
  if (!S.bookId) return;
  try {
    const data = await api('/api/tasks', {
      method: 'POST',
      body: JSON.stringify({
        book_id: S.bookId,
        force: document.getElementById('forceInput').checked,
        single_voice: document.getElementById('singleVoiceInput').checked,
      }),
    });
    toast('任务 #' + data.task_id + ' 已入队');
    await loadTasks();
  } catch (err) { toast(err.message, true); }
}

async function loadTasks() {
  if (!S.bookId) return;
  try {
    const list = await api('/api/tasks?book_id=' + encodeURIComponent(S.bookId));
    const details = await Promise.all(list.tasks.slice(0, 4).map(task =>
      api('/api/tasks/' + task.id).then(data => data.task).catch(() => task)));
    S.tasks = details;
    renderTasks();
    schedulePoll();
  } catch (err) { toast(err.message, true); }
}

function schedulePoll() {
  if (S.pollTimer) { clearTimeout(S.pollTimer); S.pollTimer = null; }
  if (S.tasks.some(task => task.status === 'running' || task.status === 'queued')) {
    S.pollTimer = setTimeout(() => refreshRunning(), 2000);
  }
}

async function refreshRunning() {
  try {
    const active = S.tasks.filter(task =>
      task.status === 'running' || task.status === 'queued' || task.status === 'paused');
    if (active.length) {
      const wasRunning = active.some(task => task.status === 'running');
      const updated = await Promise.all(active.map(task =>
        api('/api/tasks/' + task.id).then(data => data.task).catch(() => task)));
      for (const task of updated) {
        const index = S.tasks.findIndex(item => item.id === task.id);
        if (index >= 0) S.tasks[index] = task;
      }
      renderTasks();
      if (wasRunning && updated.every(task => task.status !== 'running' && task.status !== 'queued')) {
        loadOutputs();
      }
    }
  } catch (err) { /* 轮询失败静默，下一次再试 */ }
  schedulePoll();
}

function renderTasks() {
  const box = document.getElementById('taskList');
  if (!S.tasks.length) {
    box.innerHTML = '<p class="hint">还没有任务。勾选章节后点右上角“开始合成所选章节”。</p>';
    return;
  }
  box.innerHTML = S.tasks.map(task => {
    const badge = `<span class="badge ${esc(task.status)}">${STATUS_LABEL[task.status] || esc(task.status)}</span>`;
    const actions = [];
    if (task.status === 'running' || task.status === 'queued') actions.push(`<button class="ghost" data-act="pause" data-id="${task.id}">暂停</button>`);
    if (task.status === 'paused') actions.push(`<button class="ghost" data-act="resume" data-id="${task.id}">继续</button>`);
    if (task.status === 'running' || task.status === 'queued' || task.status === 'paused') actions.push(`<button class="ghost" data-act="cancel" data-id="${task.id}">取消</button>`);
    if (Number(task.failed_chapters) > 0) actions.push(`<button class="ghost" data-act="retry" data-id="${task.id}">重试失败</button>`);
    const chapters = (task.chapters || []).map(row => {
      const stages = Object.keys(STAGE_LABEL).map(key => {
        const status = row.stages[key] || 'pending';
        const percent = row.percent[key];
        const extra = key === 'tts' && row.tts_total ? ` ${row.tts_done}/${row.tts_total}` : '';
        return `<span class="stage ${esc(status)}">${STAGE_LABEL[key]}${extra}${status === 'done' ? ' ✓' : ''}</span>`;
      }).join('');
      const ttsPercent = row.percent.tts || 0;
      const detail = row.error ? `<span class="err">${esc(row.error)}</span>` : esc(row.detail || '');
      return `<div class="chapter-row">
        <div>${row.chapter_idx} · ${esc(row.title)}</div>
        <div><div class="stages">${stages}</div><div class="progress"><i style="width:${ttsPercent}%"></i></div></div>
        <div class="detail" title="${esc(row.detail || '')}">${detail}</div>
      </div>`;
    }).join('');
    return `<div class="task">
      <div class="task-head">
        <div class="task-title">任务 #${task.id} ｜ ${badge}
          <span class="hint">完成 ${task.done_chapters}/${task.total_chapters}${task.failed_chapters ? '，失败 ' + task.failed_chapters : ''}</span>
        </div>
        <div class="row gap">${actions.join('')}</div>
      </div>
      ${task.error ? `<div class="err" style="margin-bottom:8px">${esc(task.error)}</div>` : ''}
      <div class="task-chapters">${chapters || '<span class="hint">无章节</span>'}</div>
    </div>`;
  }).join('');
  box.querySelectorAll('button[data-act]').forEach(button => {
    button.addEventListener('click', () => taskAction(button.dataset.id, button.dataset.act));
  });
}

async function taskAction(taskId, action) {
  try {
    const data = await api(`/api/tasks/${taskId}/${action}`, { method: 'POST', body: '{}' });
    if (action === 'retry') toast('已创建重试任务 #' + data.task_id);
    else toast('已' + ({ pause: '暂停', resume: '继续', cancel: '取消' }[action] || action));
    await loadTasks();
  } catch (err) { toast(err.message, true); }
}

// ------------------------------------------------------------------ 成品
async function loadOutputs() {
  if (!S.bookId) return;
  try {
    const data = await api(`/api/books/${encodeURIComponent(S.bookId)}/outputs`);
    const box = document.getElementById('outputList');
    if (!data.outputs.length) {
      box.innerHTML = '<p class="hint">还没有成品。跑完一章后这里会出现 WAV / SRT / M4B。</p>';
      return;
    }
    box.innerHTML = data.outputs.map(item => `
      <div class="output">
        <div class="name">${esc(item.name)}</div>
        <div class="meta">${esc(item.kind_label)} ｜ ${fmtBytes(item.size)} ｜ ${fmtDuration(item.duration_ms)}${item.chapter_idx > 0 ? ' ｜ 第 ' + item.chapter_idx + ' 章' : ''}</div>
        ${item.kind === 'srt'
          ? `<a href="${item.download}">下载字幕</a>`
          : `<audio controls preload="none" src="${item.url}"></audio>`}
        <div style="margin-top:6px"><a href="${item.download}">下载</a></div>
      </div>`).join('');
  } catch (err) { toast(err.message, true); }
}

// ------------------------------------------------------------------ 启动
window.addEventListener('DOMContentLoaded', async () => {
  bindDropzone();
  loadHealth();
  const realAge = document.getElementById('realAgeInput');
  if (realAge) realAge.addEventListener('change', () => toggleRealAge(realAge));
  const copyBtn = document.getElementById('copyCastBtn');
  if (copyBtn) copyBtn.addEventListener('click', copyCastConfig);
  const ttsBtn = document.getElementById('saveTtsBtn');
  if (ttsBtn) ttsBtn.addEventListener('click', saveBookTts);
  await loadBooks();
  const hash = location.hash.match(/#book=(.+)/);
  if (hash) await selectBook(decodeURIComponent(hash[1]));
});
