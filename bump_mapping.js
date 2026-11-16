import { createInstance, renderInstance, shadingTypeToInt, shadowTypeToInt } from './instance.js';

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
var shared = {
    time: 0,
    paused: false,
    instances: [],
    selectedIndex: 0,
};

window.fauxpas = shared;   // debug/automation hook (see lab/shot.js)

var MAX_INSTANCES = 4;

// ---------------------------------------------------------------------------
// Label maps
// ---------------------------------------------------------------------------
var SHADING_LABELS = {
    diffuse: 'None', normal: 'Normal Map', parallax: 'Parallax',
    steep: 'Steep', pom: 'POM', contact_refinement: 'Contact', iterative: 'Iterative',
};
var SHADOW_LABELS = {
    none: 'No Shadow', hard: 'Hard POM', soft: 'Soft POM',
    fxps: 'FXPS', haps: 'HAPS', contact: 'Contact',
    binary: 'Binary Search', cone: 'Cone Traced', relief: 'Relief',
    slope: 'Slope-Guided',
    nms: 'NMS',
    horizon: 'Horizon',
};

// ---------------------------------------------------------------------------
// Cost labels
// ---------------------------------------------------------------------------
function update_cost_labels_from(N, S) {
    var shading_costs = {
        'diffuse':   '0 tex',
        'normal':    '0 tex',
        'parallax':  '1 tex',
        'steep':     '\u2264' + (1 + N) + ' tex',
        'pom':       '\u2264' + (N + 2) + ' tex',
        'iterative':      N + ' tex',
        'contact_refinement': '\u2264' + (1 + 2*N) + ' tex',
    };
    var shadow_costs = {
        'none':       '0 tex',
        'hard':       '\u2264' + (1 + S) + ' tex',
        'soft':       (1 + S) + ' tex',
        'fxps':       (1 + S) + ' tex',
        'haps':       (1 + S) + ' tex',
        'contact':    '\u2264' + (1 + S) + ' tex',
        'binary':     '\u2264' + (S + 10) + ' tex',
        'cone':       (1 + S) + ' tex',
        'relief':     '\u2264' + (S + 7) + ' tex',
        'slope':      '\u2264' + (1 + S) + ' tex',
        'nms':        (1 + S) + ' tex',
        'horizon':    S + ' tex',
    };
    for (var val in shading_costs) {
        var el = document.getElementById('cost_shading_' + val);
        if (el) el.textContent = '(' + shading_costs[val] + ')';
    }
    for (var val in shadow_costs) {
        var el = document.getElementById('cost_shadow_' + val);
        if (el) el.textContent = '(' + shadow_costs[val] + ')';
    }
}

function update_cost_labels() {
    var N = parseFloat(document.getElementById("steps").value);
    var S = parseFloat(document.getElementById("shadow_steps").value);
    update_cost_labels_from(N, S);
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function startRenderLoop() {
    setInterval(function () {
        if (!shared.paused) {
            shared.time++;
        }
        for (var i = 0; i < shared.instances.length; i++) {
            renderInstance(shared.instances[i], shared.time);
        }
    }, 15);
}

// ---------------------------------------------------------------------------
// Title bar
// ---------------------------------------------------------------------------
function updateTitleBar(inst) {
    var s = inst.settings;
    var type = shadingTypeToInt(s.shading_type);
    var shadow = shadowTypeToInt(s.shadow_type);
    var shadingLabel = (SHADING_LABELS[s.shading_type] || s.shading_type)
        + (type >= 3 ? ' (n=' + s.steps + ')' : '');
    var shadowLabel = (SHADOW_LABELS[s.shadow_type] || s.shadow_type)
        + (shadow > 0 ? ' (n=' + s.shadow_steps + ')': '');
    inst.titleTextEl.textContent = shadingLabel + ', ' + shadowLabel;
}

// ---------------------------------------------------------------------------
// Selection & control sync
// ---------------------------------------------------------------------------
function selectInstance(index) {
    shared.selectedIndex = index;
    for (var i = 0; i < shared.instances.length; i++) {
        var cls = shared.instances[i].titleBarEl.classList;
        if (i === index) cls.add('selected'); else cls.remove('selected');
    }
    syncControlsFromInstance(shared.instances[index]);
}

function syncControlsFromInstance(inst) {
    var s = inst.settings;

    // Texture set radios
    var texNames = ['bump', 'bricks', 'rocks'];
    var texRadio = document.querySelector('input[name="tex_set"][value="' + texNames[s.tex_set] + '"]');
    if (texRadio) texRadio.checked = true;

    // Show tex checkbox
    document.getElementById('show_tex').checked = s.show_tex;

    // Shading type radio
    var shadingRadio = document.querySelector('input[name="shading_type"][value="' + s.shading_type + '"]');
    if (shadingRadio) shadingRadio.checked = true;

    // Shadow type radio
    var shadowRadio = document.querySelector('input[name="shadow_type"][value="' + s.shadow_type + '"]');
    if (shadowRadio) shadowRadio.checked = true;

    // Sliders + display spans
    document.getElementById('scale').value = s.scale;
    document.getElementById('scale_val').textContent = (0.01 * s.scale).toFixed(2);

    document.getElementById('bias_ratio').value = s.bias_ratio;
    document.getElementById('bias_ratio_val').textContent = Math.round(s.bias_ratio) + '%';

    document.getElementById('steps').value = s.steps;
    document.getElementById('steps_val').textContent = s.steps;

    document.getElementById('shadow_steps').value = s.shadow_steps;
    document.getElementById('shadow_steps_val').textContent = s.shadow_steps;

    document.getElementById('fxps_alpha').value = s.fxps_alpha;
    document.getElementById('fxps_alpha_val').textContent = parseFloat(s.fxps_alpha).toFixed(1);

    document.getElementById('light_radius').value = s.light_radius;
    document.getElementById('light_radius_val').textContent = parseFloat(s.light_radius).toFixed(2);

    document.getElementById('prefilter').value = s.prefilter;
    document.getElementById('prefilter_val').textContent = parseFloat(s.prefilter).toFixed(2);

    document.getElementById('haps_k').value = s.haps_k;
    document.getElementById('haps_k_val').textContent = parseFloat(s.haps_k).toFixed(2);

    // Conditional visibility
    var type = shadingTypeToInt(s.shading_type);
    document.getElementById('scale_control').style.visibility = type >= 2 ? 'visible' : 'hidden';
    document.getElementById('bias_ratio_control').style.visibility = type == 5 ? 'visible' : 'hidden';
    document.getElementById('step_control').style.visibility = type >= 3 ? 'visible' : 'hidden';

    var shadow = shadowTypeToInt(s.shadow_type);
    document.getElementById('shadow_step_control').style.visibility = shadow > 0 ? 'visible' : 'hidden';
    document.getElementById('fxps_alpha_control').style.visibility = (shadow == 3 || shadow == 8) ? 'visible' : 'hidden';
    document.getElementById('light_radius_control').style.visibility = shadow == 11 ? 'visible' : 'hidden';
    document.getElementById('prefilter_control').style.visibility = shadow == 11 ? 'visible' : 'hidden';
    document.getElementById('haps_k_control').style.visibility = shadow == 3 ? 'visible' : 'hidden';

    // Cost labels
    update_cost_labels_from(s.steps, s.shadow_steps);
}

// ---------------------------------------------------------------------------
// Control event binding (once, writes to selected instance)
// ---------------------------------------------------------------------------
function bindControlEvents() {
    function sel() { return shared.instances[shared.selectedIndex]; }

    // Texture set radios
    var texSetMap = { bump: 0, bricks: 1, rocks: 2 };
    var texRadios = document.querySelectorAll('input[name="tex_set"]');
    for (var i = 0; i < texRadios.length; i++) {
        texRadios[i].addEventListener('change', function () {
            sel().settings.tex_set = texSetMap[this.value];
        });
    }

    // Show tex checkbox
    document.getElementById('show_tex').addEventListener('change', function () {
        sel().settings.show_tex = this.checked;
    });

    // Shading type radios
    var shadingRadios = document.querySelectorAll('input[name="shading_type"]');
    for (var i = 0; i < shadingRadios.length; i++) {
        shadingRadios[i].addEventListener('change', function () {
            sel().settings.shading_type = this.value;
            var type = shadingTypeToInt(this.value);
            document.getElementById('scale_control').style.visibility = type >= 2 ? 'visible' : 'hidden';
            document.getElementById('bias_ratio_control').style.visibility = type == 5 ? 'visible' : 'hidden';
            document.getElementById('step_control').style.visibility = type >= 3 ? 'visible' : 'hidden';
            updateTitleBar(sel());
        });
    }

    // Shadow type radios
    var shadowRadios = document.querySelectorAll('input[name="shadow_type"]');
    for (var i = 0; i < shadowRadios.length; i++) {
        shadowRadios[i].addEventListener('change', function () {
            sel().settings.shadow_type = this.value;
            var shadow = shadowTypeToInt(this.value);
            document.getElementById('shadow_step_control').style.visibility = shadow > 0 ? 'visible' : 'hidden';
            document.getElementById('fxps_alpha_control').style.visibility = (shadow == 3 || shadow == 8) ? 'visible' : 'hidden';
            document.getElementById('light_radius_control').style.visibility = shadow == 11 ? 'visible' : 'hidden';
            document.getElementById('prefilter_control').style.visibility = shadow == 11 ? 'visible' : 'hidden';
            document.getElementById('haps_k_control').style.visibility = shadow == 3 ? 'visible' : 'hidden';
            updateTitleBar(sel());
        });
    }

    // Sliders
    document.getElementById('scale').addEventListener('input', function () {
        sel().settings.scale = parseFloat(this.value);
        document.getElementById('scale_val').textContent = (0.01 * sel().settings.scale).toFixed(2);
    });
    document.getElementById('bias_ratio').addEventListener('input', function () {
        sel().settings.bias_ratio = parseFloat(this.value);
        document.getElementById('bias_ratio_val').textContent = Math.round(sel().settings.bias_ratio) + '%';
    });
    document.getElementById('steps').addEventListener('input', function () {
        sel().settings.steps = parseFloat(this.value);
        document.getElementById('steps_val').textContent = sel().settings.steps;
        update_cost_labels();
        updateTitleBar(sel());
    });
    document.getElementById('shadow_steps').addEventListener('input', function () {
        sel().settings.shadow_steps = parseFloat(this.value);
        document.getElementById('shadow_steps_val').textContent = sel().settings.shadow_steps;
        update_cost_labels();
        updateTitleBar(sel());
    });
    document.getElementById('fxps_alpha').addEventListener('input', function () {
        sel().settings.fxps_alpha = parseFloat(this.value);
        document.getElementById('fxps_alpha_val').textContent = sel().settings.fxps_alpha.toFixed(1);
    });
    document.getElementById('light_radius').addEventListener('input', function () {
        sel().settings.light_radius = parseFloat(this.value);
        document.getElementById('light_radius_val').textContent = sel().settings.light_radius.toFixed(2);
    });
    document.getElementById('prefilter').addEventListener('input', function () {
        sel().settings.prefilter = parseFloat(this.value);
        document.getElementById('prefilter_val').textContent = sel().settings.prefilter.toFixed(2);
    });
    document.getElementById('haps_k').addEventListener('input', function () {
        sel().settings.haps_k = parseFloat(this.value);
        document.getElementById('haps_k_val').textContent = sel().settings.haps_k.toFixed(2);
    });
}

// ---------------------------------------------------------------------------
// Add / remove instances
// ---------------------------------------------------------------------------
function addInstance() {
    if (shared.instances.length >= MAX_INSTANCES) return;

    var strip = document.getElementById('canvas_strip');
    var addBtn = document.getElementById('add_canvas_btn');

    // Container
    var container = document.createElement('div');
    container.className = 'canvas-container';

    // Title bar
    var titleBar = document.createElement('div');
    titleBar.className = 'canvas-title-bar';
    var titleText = document.createElement('span');
    titleText.className = 'title-text';
    titleBar.appendChild(titleText);

    // Close button (hidden if only one instance — managed in removeInstance)
    var closeBtn = document.createElement('button');
    closeBtn.className = 'canvas-close-btn';
    closeBtn.textContent = '\u00d7';
    closeBtn.title = 'Remove';
    titleBar.appendChild(closeBtn);
    container.appendChild(titleBar);

    // Canvas
    var canvasEl = document.createElement('canvas');
    canvasEl.width = 2;   // will be synced to display size on first render
    canvasEl.height = 2;
    container.appendChild(canvasEl);

    // Stats overlay
    var statsEl = document.createElement('div');
    statsEl.className = 'stats-overlay';
    container.appendChild(statsEl);

    strip.insertBefore(container, addBtn);

    var inst = createInstance(canvasEl);
    if (!inst) {
        container.remove();
        return;
    }

    inst.containerEl = container;
    inst.titleBarEl = titleBar;
    inst.titleTextEl = titleText;
    inst.closeBtnEl = closeBtn;
    inst.statsOverlayEl = statsEl;

    shared.instances.push(inst);

    // Wire events — close over inst object, not index
    titleBar.addEventListener('click', function (e) {
        if (e.target === closeBtn) return;
        selectInstance(shared.instances.indexOf(inst));
    });
    canvasEl.addEventListener('click', function () {
        shared.paused = !shared.paused;
    });
    closeBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        removeInstance(shared.instances.indexOf(inst));
    });

    updateTitleBar(inst);
    selectInstance(shared.instances.indexOf(inst));
    updateCloseButtons();
    updateAddButton();
}

function removeInstance(index) {
    if (index < 0 || shared.instances.length <= 1) return;
    var inst = shared.instances[index];
    inst.containerEl.remove();
    shared.instances.splice(index, 1);

    var newSel = Math.min(shared.selectedIndex, shared.instances.length - 1);
    selectInstance(newSel);
    updateCloseButtons();
    updateAddButton();
}

function updateCloseButtons() {
    var hide = shared.instances.length <= 1;
    for (var i = 0; i < shared.instances.length; i++) {
        shared.instances[i].closeBtnEl.style.display = hide ? 'none' : '';
    }
}

function updateAddButton() {
    var btn = document.getElementById('add_canvas_btn');
    btn.style.display = shared.instances.length >= MAX_INSTANCES ? 'none' : '';
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function initBumpMapping() {
    document.getElementById('add_canvas_btn').addEventListener('click', function () {
        addInstance();
    });
    bindControlEvents();
    addInstance();
    startRenderLoop();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBumpMapping);
} else {
    initBumpMapping();
}
