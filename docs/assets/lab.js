// 图表的按需装载与绘制。
// 之所以用脚本注入而不是 fetch：file:// 下 fetch 会被 CORS 拦，脚本标签不会，
// 因此这个站点双击打开也能正常工作。
window.__LAB_FIGS__ = window.__LAB_FIGS__ || {};
window.__LAB_LOADED__ = window.__LAB_LOADED__ || {};

function labDraw() {
  Object.keys(window.__LAB_FIGS__).forEach(function (id) {
    var el = document.getElementById(id);
    if (!el || el.dataset.labDrawn === "1") return;
    // 容器不可见（档位未选中）时不画：Plotly 在 display:none 的容器里量不到尺寸
    if (el.offsetParent === null) return;
    Plotly.newPlot(id, window.__LAB_FIGS__[id].data, window.__LAB_FIGS__[id].layout,
                   {displaylogo: false, responsive: true});
    el.dataset.labDrawn = "1";
  });
}

function labLoad(key, path) {
  if (window.__LAB_LOADED__[key]) { labDraw(); return; }
  window.__LAB_LOADED__[key] = true;
  var script = document.createElement("script");
  script.src = path;
  script.onload = labDraw;
  script.onerror = function () { console.error("图表数据加载失败：" + path); };
  document.head.appendChild(script);
}

// 首屏只装载当前选中档位的数据；切换档位时按需再装载。
function labInitTabs() {
  var inputs = document.querySelectorAll('input[name="gear"]');
  inputs.forEach(function (input) {
    input.addEventListener("change", function () {
      if (input.checked) labLoad(input.dataset.key, input.dataset.src);
      // 之前因不可见而跳过的图，会在切回该档时补画
      window.setTimeout(labDraw, 0);
    });
  });
  var checked = document.querySelector('input[name="gear"]:checked') || inputs[0];
  if (checked) labLoad(checked.dataset.key, checked.dataset.src);
}

// 独立组合页：容器本来就是可见的，直接装载即可。
function labInitSingle(key, path) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { labLoad(key, path); });
  } else {
    labLoad(key, path);
  }
}
