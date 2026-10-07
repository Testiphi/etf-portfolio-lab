// 网络与 Plotly 故障的确定性测试，不依赖行情服务或浏览器。
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const runtime = fs.readFileSync(path.join(__dirname, '../src/etf_lab/reports/assets/lab.js'), 'utf8');

function fixture() {
  const scripts = [], timers = new Map(), boxes = [];
  let timerId = 0, draws = 0;
  function element() {
    return {dataset: {}, children: [], hidden: true, offsetParent: {},
      appendChild(child) { this.children.push(child); },
      replaceChildren() { this.children = []; },
      remove() { this.removed = true; }};
  }
  const chart = element(), box = element();
  box.dataset.chartKey = 'one'; boxes.push(box);
  const document = {
    head: {appendChild(script) { scripts.push(script); }},
    createElement: element,
    querySelectorAll: () => boxes,
    querySelector: () => ({src: 'assets/plotly.min.js'}),
    getElementById: id => id === 'one-fig-nav' ? chart : null,
  };
  const context = {document, console, Promise,
    setTimeout(fn) { timers.set(++timerId, fn); return timerId; },
    clearTimeout(id) { timers.delete(id); },
    Plotly: {newPlot() { draws++; return Promise.resolve(); }},
  };
  context.window = context;
  vm.createContext(context); vm.runInContext(runtime, context);
  const data = () => { context.__LAB_FIGS__['one-fig-nav'] = {data: [], layout: {}}; };
  return {context, scripts, timers, chart, box, data, draws: () => draws};
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('failed download stays retryable; duplicate requests and successful draws are reused', async () => {
  const f = fixture();
  const first = f.context.labLoad('one', 'one.js');
  assert.equal(f.context.labLoad('one', 'one.js'), first);
  await tick(); assert.equal(f.scripts.length, 1);
  f.scripts[0].onerror(); await first;
  assert.equal(f.context.__LAB_LOADED__.one, undefined);
  assert.equal(f.box.hidden, false);
  assert.equal(f.box.children[1].textContent, '重试图表');
  f.box.children[1].onclick(); await tick();
  const retry = f.context.labPending.one;
  f.data(); f.scripts[1].onload(); await retry;
  assert.equal(f.box.hidden, true); assert.equal(f.draws(), 1);
  await f.context.labLoad('one', 'one.js');
  assert.equal(f.scripts.length, 2); assert.equal(f.draws(), 1);
});

test('script timeout clears pending state and removes failed script', async () => {
  const f = fixture(), pending = f.context.labLoad('one', 'one.js');
  await tick(); [...f.timers.values()][0](); await pending;
  assert.equal(f.scripts[0].removed, true);
  assert.equal(f.context.labPending.one, undefined);
  assert.equal(f.context.__LAB_LOADED__.one, undefined);
});

test('missing Plotly can fail then recover without reloading the page', async () => {
  const f = fixture(); delete f.context.Plotly;
  const first = f.context.labLoad('one', 'one.js');
  await tick(); assert.equal(f.scripts[0].src, 'assets/plotly.min.js');
  f.scripts[0].onerror(); await first;
  const retry = f.context.labLoad('one', 'one.js');
  await tick();
  f.context.Plotly = {newPlot: () => Promise.resolve()};
  f.scripts[1].onload(); await tick();
  f.data(); f.scripts[2].onload(); await retry;
  assert.equal(f.chart.dataset.labDrawn, '1');
});

test('render rejection retries drawing without refetching data', async () => {
  const f = fixture();
  f.context.Plotly.newPlot = () => Promise.reject(new Error('render'));
  const first = f.context.labLoad('one', 'one.js');
  await tick(); f.data(); f.scripts[0].onload(); await first;
  assert.equal(f.chart.dataset.labDrawn, undefined);
  f.context.Plotly.newPlot = () => Promise.resolve();
  await f.context.labLoad('one', 'one.js');
  assert.equal(f.scripts.length, 1); assert.equal(f.chart.dataset.labDrawn, '1');
});

test('hidden charts are drawn on returning to their preset', async () => {
  const f = fixture(); f.chart.offsetParent = null;
  const pending = f.context.labLoad('one', 'one.js');
  await tick(); f.data(); f.scripts[0].onload(); await pending;
  assert.equal(f.draws(), 0);
  f.chart.offsetParent = {};
  await f.context.labLoad('one', 'one.js');
  assert.equal(f.draws(), 1);
});

test('an empty data script is not marked loaded', async () => {
  const f = fixture(), pending = f.context.labLoad('one', 'one.js');
  await tick(); f.scripts[0].onload(); await pending;
  assert.equal(f.context.__LAB_LOADED__.one, undefined);
  assert.equal(f.box.hidden, false);
});
