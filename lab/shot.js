// Screenshot the live WebGL demo. Usage:
//   node lab/shot.js out.png '[{"tex":"rocks","shading":"iterative","shadow":"fxps","steps":4,"shadow_steps":4}, ...]' [time] [scale]
// Serves the repo root itself; one canvas per config, frozen at `time`.
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.png': 'image/png' };
(async () => {
  const [out, cfgJson, time = '40', scale = '8'] = process.argv.slice(2);
  const cfgs = JSON.parse(cfgJson);
  const srv = http.createServer((q, r) => {
    const f = path.join(root, decodeURIComponent(q.url.split('?')[0]));
    fs.readFile(f, (e, d) => e ? (r.writeHead(404), r.end()) : (r.writeHead(200, { 'Content-Type': mime[path.extname(f)] || 'text/plain' }), r.end(d)));
  }).listen(0);
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 420 * cfgs.length + 80, height: 1000 }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error') console.log('console:', m.text()); });
  page.on('dialog', d => { console.log('ALERT:', d.message()); d.dismiss(); });
  await page.goto(`http://localhost:${srv.address().port}/index.html`);
  const set = async (id, v) => page.$eval('#' + id, (el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }, v);
  for (let i = 0; i < cfgs.length; i++) {
    if (i > 0) await page.click('#add_canvas_btn');
    const c = cfgs[i];
    await page.check(`input[name="tex_set"][value="${c.tex || 'bump'}"]`);
    await page.check(`input[name="shading_type"][value="${c.shading || 'iterative'}"]`);
    await page.check(`input[name="shadow_type"][value="${c.shadow || 'none'}"]`);
    await set('scale', c.scale ?? scale);
    await set('steps', c.steps ?? 4);
    if ((c.shadow || 'none') !== 'none') await set('shadow_steps', c.shadow_steps ?? 4);
    if (c.prefilter !== undefined) await set('prefilter', c.prefilter);
    if (c.light_radius !== undefined) await set('light_radius', c.light_radius);
    if (c.show_tex === false) await page.uncheck('#show_tex');
  }
  await page.evaluate(t => { window.fauxpas.paused = true; window.fauxpas.time = t; }, parseFloat(time));
  await page.waitForTimeout(1500);
  await page.locator('#canvas_strip, .canvas-strip').first().screenshot({ path: out, timeout: 120000 }).catch(() => page.screenshot({ path: out }));
  await browser.close(); srv.close();
})();
