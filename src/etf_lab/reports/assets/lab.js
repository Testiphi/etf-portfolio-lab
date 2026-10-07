// 使用脚本文件保留 file:// 离线支持；加载状态只在成功后记为 loaded。
window.__LAB_FIGS__ = window.__LAB_FIGS__ || {};
window.__LAB_LOADED__ = window.__LAB_LOADED__ || {};
var labPending = {};
var labPlotlyPending = null;

function labStatus(key, message, retry) {
  document.querySelectorAll('[data-chart-key]').forEach(function (box) {
    if (box.dataset.chartKey !== key) return;
    box.hidden = !message;
    box.replaceChildren();
    var text = document.createElement('span');
    text.textContent = message;
    box.appendChild(text);
    if (retry) {
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = '重试图表';
      button.onclick = retry;
      box.appendChild(button);
    }
  });
}

function labScript(path) {
  return new Promise(function (resolve, reject) {
    var script = document.createElement('script');
    var timer = window.setTimeout(function () { finish(new Error('加载超时')); }, 20000);
    function finish(error) {
      window.clearTimeout(timer);
      script.onload = script.onerror = null;
      if (error) { script.remove(); reject(error); }
      else resolve();
    }
    script.src = path;
    script.onload = function () { finish(); };
    script.onerror = function () { finish(new Error('加载失败')); };
    document.head.appendChild(script);
  });
}

function labEnsurePlotly() {
  if (window.Plotly && typeof window.Plotly.newPlot === 'function') return Promise.resolve();
  if (!labPlotlyPending) {
    var source = document.querySelector('script[data-plotly]');
    labPlotlyPending = (source ? labScript(source.src) : Promise.reject(new Error('缺少图表库路径')))
      .then(function () {
        if (!window.Plotly || typeof window.Plotly.newPlot !== 'function') throw new Error('图表库不可用');
      }).finally(function () { labPlotlyPending = null; });
  }
  return labPlotlyPending;
}

async function labDraw(key) {
  var jobs = Object.keys(window.__LAB_FIGS__).filter(function (id) {
    return id.indexOf(key + '-') === 0;
  }).map(async function (id) {
    var el = document.getElementById(id);
    if (!el || el.offsetParent === null || el.dataset.labDrawn === '1') return;
    try {
      await window.Plotly.newPlot(el, window.__LAB_FIGS__[id].data,
        window.__LAB_FIGS__[id].layout, {displaylogo: false, responsive: true});
      el.dataset.labDrawn = '1';
    } catch (error) {
      delete el.dataset.labDrawn;
      throw error;
    }
  });
  // 等待所有图完成再开放重试，避免失败的一张导致其它图重复绘制。
  var outcomes = await Promise.allSettled(jobs);
  if (outcomes.some(function (item) { return item.status === 'rejected'; })) {
    throw new Error('部分图表绘制失败');
  }
}

function labLoad(key, path) {
  if (labPending[key]) return labPending[key];
  labStatus(key, '正在加载图表…');
  labPending[key] = (async function () {
    try {
      await labEnsurePlotly();
      if (!window.__LAB_LOADED__[key]) {
        await labScript(path);
        if (!Object.keys(window.__LAB_FIGS__).some(function (id) { return id.indexOf(key + '-') === 0; })) {
          throw new Error('图表数据为空');
        }
        window.__LAB_LOADED__[key] = true;
      }
      await labDraw(key);
      labStatus(key, '');
    } catch (error) {
      labStatus(key, '图表未能完成加载，请检查网络或本地文件后重试。', function () { labLoad(key, path); });
    } finally {
      delete labPending[key];
    }
  })();
  return labPending[key];
}

function labInitTabs() {
  var inputs = document.querySelectorAll('input[name="gear"]');
  inputs.forEach(function (input) {
    input.addEventListener('change', function () {
      if (input.checked) labLoad(input.dataset.key, input.dataset.src);
    });
  });
  var checked = document.querySelector('input[name="gear"]:checked') || inputs[0];
  if (checked) labLoad(checked.dataset.key, checked.dataset.src);
}

function labInitSingle(key, path) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { labLoad(key, path); });
  } else {
    labLoad(key, path);
  }
}
