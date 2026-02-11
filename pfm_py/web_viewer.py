"""
Generate a web-based interactive 3D viewer (HTML + JSON assets) for full/partial meshes
with matched colors. Uses Three.js (CDN) with OrbitControls to enable rotation/zoom/pan.

Outputs under a given directory:
- full.json: vertices/faces/colors for full mesh (continuous colors)
- partial_method.json: partial mesh colored by method matches
- partial_gt.json: partial mesh colored by GT matches (optional)
- interactive_view.html: self-contained page loading above JSON assets
"""

import os
import json
import urllib.request
import numpy as np
import open3d as o3d


def _hsv_colormap(n: int) -> np.ndarray:
    colors = np.zeros((n, 3), dtype=float)
    for i in range(n):
        h = i / max(1, n)
        colors[i] = [abs(1 - 2*abs(h - 0.0)), abs(1 - 2*abs(h - 0.33)), abs(1 - 2*abs(h - 0.66))]
    return np.clip(colors, 0.0, 1.0)


def _mesh_to_json_dict(verts: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    return {
        "vertices": verts.tolist(),
        "faces": faces.astype(int).tolist(),
        "colors": colors.tolist(),
    }


def _write_json(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def _ensure_three_js(out_dir: str) -> str:
  """Download three.min.js locally into out_dir (if missing). Returns the local path."""
  js_path = os.path.join(out_dir, 'three.min.js')
  if os.path.exists(js_path) and os.path.getsize(js_path) > 100_000:
    return js_path
  urls = [
    'https://raw.githubusercontent.com/mrdoob/three.js/r150/build/three.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r150/three.min.js'
  ]
  for url in urls:
    try:
      with urllib.request.urlopen(url, timeout=10) as resp:
        content = resp.read()
      if len(content) > 100_000:
        with open(js_path, 'wb') as f:
          f.write(content)
        return js_path
    except Exception:
      continue
  # Fallback: write a stub that will cause THREE to be undefined (our HTML handles absence gracefully by not crashing)
  with open(js_path, 'w', encoding='utf-8') as f:
    f.write('// three.min.js download failed; viewer will attempt fallback controls')
  return js_path


def _write_html(path: str, has_gt: bool):
    # Precompute optional blocks and simple tokens (avoid f-strings entirely)
    toggle_btn_html = '<button id="toggleBtn" style="margin-left:12px;">Toggle GT / Method</button>' if has_gt else ''
    partial_label = 'Method / GT' if has_gt else 'Method'
    partial_gt_loader = 'await loadJSON("partial_gt.json")' if has_gt else 'null'
    gt_init_block = (
        """
      pgMesh = buildMesh(partialGT);
      pgMesh.name = 'partial_gt';
      right.scene.add(pgMesh);
      pgMesh.visible = false;
        """
    ) if has_gt else ''
    toggle_handler = (
        """
      document.getElementById('toggleBtn').addEventListener('click', () => {
        if (!pgMesh) return;
        const showGT = !pgMesh.visible;
        pgMesh.visible = showGT;
        pmMesh.visible = !showGT;
      });
        """
    ) if has_gt else ''

    # Base HTML with simple placeholder tokens
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Interactive Mesh Viewer</title>
  <style>
    html {{ height: 100%; }}
    body {{ height: 100%; }}
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #toolbar {{ padding: 8px; background: #f4f4f4; border-bottom: 1px solid #ddd; }}
    #container {{ display: flex; height: calc(100vh - 42px); min-height: 60vh; }}
    .panel {{ flex: 1; position: relative; min-height: 300px; }}
    #left, #right {{ width: 100%; height: 100%; }}
    canvas {{ display: block; width: 100%; height: 100%; }}
    .label {{ position: absolute; left: 8px; top: 8px; background: rgba(255,255,255,0.8); padding: 4px 8px; border-radius: 4px; border: 1px solid #ddd; }}
  </style>
  <script src="./three.min.js"></script>
  <script src="./OrbitControls.js"></script>
  <script>
    // Inline fallback OrbitControls for offline usage
    if (!window.THREE || !THREE.OrbitControls) {
      (function(){
        if (!window.THREE) return;
        THREE.OrbitControls = function(camera, dom) {
          const target = new THREE.Vector3();
          const spherical = new THREE.Spherical();
          const tmp = new THREE.Vector3();
          function sync(){ tmp.copy(camera.position).sub(target); spherical.setFromVector3(tmp); }
          sync();
          let dragging=false, panning=false, lastX=0, lastY=0;
          dom.addEventListener('contextmenu', e=>e.preventDefault());
          dom.addEventListener('mousedown', e=>{ if(e.button===0) dragging=true; else panning=true; lastX=e.clientX; lastY=e.clientY; });
          window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
          dom.addEventListener('mousemove', e=>{
            const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;
            if(dragging){
              spherical.theta -= dx*0.005; spherical.phi -= dy*0.005; spherical.phi=Math.max(0.001,Math.min(Math.PI-0.001,spherical.phi));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            } else if(panning){
              const panSpeed = spherical.radius*0.001; const forward=camera.getWorldDirection(new THREE.Vector3());
              const right=new THREE.Vector3().crossVectors(forward,camera.up).normalize(); const up=camera.up.clone().normalize();
              target.add(right.multiplyScalar(-dx*panSpeed)).add(up.multiplyScalar(dy*panSpeed));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            }
          });
          dom.addEventListener('wheel', e=>{ e.preventDefault(); const s=Math.pow(1.1, Math.sign(e.deltaY)); spherical.radius=Math.max(0.001, spherical.radius*s);
            tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
          }, { passive:false });
          this.target = target; this.update = function(){};
        };
      })();
    }
  </script>
  <script>
    // Inline fallback OrbitControls for offline usage
    if (!window.THREE || !THREE.OrbitControls) {
      (function(){
        if (!window.THREE) return;
        THREE.OrbitControls = function(camera, dom) {
          const target = new THREE.Vector3();
          const spherical = new THREE.Spherical();
          const tmp = new THREE.Vector3();
          function sync(){ tmp.copy(camera.position).sub(target); spherical.setFromVector3(tmp); }
          sync();
          let dragging=false, panning=false, lastX=0, lastY=0;
          dom.addEventListener('contextmenu', e=>e.preventDefault());
          dom.addEventListener('mousedown', e=>{ if(e.button===0) dragging=true; else panning=true; lastX=e.clientX; lastY=e.clientY; });
          window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
          dom.addEventListener('mousemove', e=>{
            const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;
            if(dragging){
              spherical.theta -= dx*0.005; spherical.phi -= dy*0.005; spherical.phi=Math.max(0.001,Math.min(Math.PI-0.001,spherical.phi));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            } else if(panning){
              const panSpeed = spherical.radius*0.001; const forward=camera.getWorldDirection(new THREE.Vector3());
              const right=new THREE.Vector3().crossVectors(forward,camera.up).normalize(); const up=camera.up.clone().normalize();
              target.add(right.multiplyScalar(-dx*panSpeed)).add(up.multiplyScalar(dy*panSpeed));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            }
          });
          dom.addEventListener('wheel', e=>{ e.preventDefault(); const s=Math.pow(1.1, Math.sign(e.deltaY)); spherical.radius=Math.max(0.001, spherical.radius*s);
            tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
          }, { passive:false });
          this.target = target; this.update = function(){};
        };
      })();
    }
  </script>
</head>
<body>
  <div id="toolbar">
    <strong>Interactive Viewer:</strong>
    <span style="margin-left:12px;">Left: Full (continuous)</span>
    <span style="margin-left:12px;">Right: Partial (__PARTIAL_LABEL__)</span>
    __TOGGLE_BTN__
  </div>
  <div id="container">
    <div class="panel"><div class="label">Full Mesh (continuous)</div><div id="left"></div></div>
    <div class="panel"><div class="label">Partial Mesh (__PARTIAL_LABEL__)</div><div id="right"></div></div>
  </div>
  <script>
    async function loadJSON(url) {{
      const res = await fetch(url);
      return await res.json();
    }}

    function makeScene(dom) {{
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xF9FAFB);
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      dom.appendChild(renderer.domElement);
      const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      const light = new THREE.DirectionalLight(0xffffff, 0.9);
      light.position.set(1,1,1);
      scene.add(light);
      const amb = new THREE.AmbientLight(0xffffff, 0.3);
      scene.add(amb);
      function onResize() {{
        const parent = dom.parentElement || dom;
        let rect = parent.getBoundingClientRect();
        let w = rect.width;
        let h = rect.height;
        if (!w || !h) {{
          w = parent.clientWidth || window.innerWidth / 2;
          h = parent.clientHeight || Math.max(300, window.innerHeight - 42);
        }}
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      }}
      window.addEventListener('resize', onResize);
      onResize();
      return {{ scene, renderer, camera, controls }};
    }}

    function buildMesh(data) {{
      const verts = data.vertices;
      const faces = data.faces;
      const cols = data.colors;
      const geom = new THREE.BufferGeometry();
      const position = new Float32Array(verts.length * 3);
      const color = new Float32Array(verts.length * 3);
      for (let i=0;i<verts.length;i++) {{
        position[3*i+0] = verts[i][0];
        position[3*i+1] = verts[i][1];
        position[3*i+2] = verts[i][2];
        color[3*i+0] = cols[i][0];
        color[3*i+1] = cols[i][1];
        color[3*i+2] = cols[i][2];
      }}
      const index = new Uint32Array(faces.length * 3);
      for (let f=0; f<faces.length; f++) {{
        index[3*f+0] = faces[f][0];
        index[3*f+1] = faces[f][1];
        index[3*f+2] = faces[f][2];
      }}
      geom.setAttribute('position', new THREE.BufferAttribute(position, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(color, 3));
      geom.setIndex(new THREE.BufferAttribute(index, 1));
      geom.computeVertexNormals();
      const mat = new THREE.MeshPhongMaterial({{ vertexColors: true, side: THREE.DoubleSide }});
      return new THREE.Mesh(geom, mat);
    }}

    async function init() {{
      const left = makeScene(document.getElementById('left'));
      const right = makeScene(document.getElementById('right'));

      const full = await loadJSON('full.json');
      const partialMethod = await loadJSON('partial_method.json');
      const partialGT = __PARTIAL_GT_LOADER__;

      const fullMesh = buildMesh(full);
      left.scene.add(fullMesh);
      const pmMesh = buildMesh(partialMethod);
      pmMesh.name = 'partial_method';
      right.scene.add(pmMesh);
      let pgMesh = null;
      __GT_INIT_BLOCK__

      function fitCamera(sceneObj, mesh) {{
        const box = new THREE.Box3().setFromObject(mesh);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const dist = maxDim * 1.5;
        sceneObj.camera.position.copy(center.clone().add(new THREE.Vector3(dist, dist, dist)));
        sceneObj.controls.target.copy(center);
        sceneObj.controls.update();
      }}

      fitCamera(left, fullMesh);
      fitCamera(right, pmMesh);

      function animate() {{
        left.renderer.render(left.scene, left.camera);
        right.renderer.render(right.scene, right.camera);
        requestAnimationFrame(animate);
      }}
      animate();

      __TOGGLE_HANDLER__
    }

    init();
  </script>
</body>
</html>
"""
    # Replace placeholders
    html = html.replace('__TOGGLE_BTN__', toggle_btn_html)
    html = html.replace('__PARTIAL_LABEL__', partial_label)
    html = html.replace('__PARTIAL_GT_LOADER__', partial_gt_loader)
    html = html.replace('__GT_INIT_BLOCK__', gt_init_block)
    html = html.replace('__TOGGLE_HANDLER__', toggle_handler)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def _write_html_embedded(path: str, has_gt: bool, full_dict: dict, method_dict: dict, gt_dict: dict | None):
    # Prepare tokens
    toggle_btn_html = '<button id="toggleBtn" style="margin-left:12px;">Toggle GT / Method</button>' if has_gt else ''
    partial_label = 'Method / GT' if has_gt else 'Method'
    gt_script = ("<script id=\"partial-gt-json\" type=\"application/json\">" + json.dumps(gt_dict) + "</script>") if has_gt and gt_dict is not None else ''
    toggle_handler = (
        """
      document.getElementById('toggleBtn').addEventListener('click', () => {
        if (!pgMesh) return;
        const showGT = !pgMesh.visible;
        pgMesh.visible = showGT;
        pmMesh.visible = !showGT;
      });
        """
    ) if has_gt else ''

    # Base HTML (no fetch; data embedded as JSON script tags)
    html = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Interactive Mesh Viewer</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; }
    #toolbar { padding: 8px; background: #f4f4f4; border-bottom: 1px solid #ddd; }
    #container { display: flex; height: calc(100vh - 42px); }
    .panel { flex: 1; position: relative; }
    #left, #right { width: 100%; height: 100%; }
    canvas { display: block; width: 100%; height: 100%; }
    .label { position: absolute; left: 8px; top: 8px; background: rgba(255,255,255,0.8); padding: 4px 8px; border-radius: 4px; border: 1px solid #ddd; }
  </style>
  <script src=\"./three.min.js\"></script>
  <script src=\"./OrbitControls.js\"></script>
  <script>
    if (!window.THREE || !THREE.OrbitControls) {
      (function(){
        if (!window.THREE) return;
        THREE.OrbitControls = function(camera, dom) {
          const target = new THREE.Vector3();
          const spherical = new THREE.Spherical();
          const tmp = new THREE.Vector3();
          function sync(){ tmp.copy(camera.position).sub(target); spherical.setFromVector3(tmp); }
          sync();
          let dragging=false, panning=false, lastX=0, lastY=0;
          dom.addEventListener('contextmenu', e=>e.preventDefault());
          dom.addEventListener('mousedown', e=>{ if(e.button===0) dragging=true; else panning=true; lastX=e.clientX; lastY=e.clientY; });
          window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
          dom.addEventListener('mousemove', e=>{
            const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;
            if(dragging){
              spherical.theta -= dx*0.005; spherical.phi -= dy*0.005; spherical.phi=Math.max(0.001,Math.min(Math.PI-0.001,spherical.phi));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            } else if(panning){
              const panSpeed = spherical.radius*0.001; const forward=camera.getWorldDirection(new THREE.Vector3());
              const right=new THREE.Vector3().crossVectors(forward,camera.up).normalize(); const up=camera.up.clone().normalize();
              target.add(right.multiplyScalar(-dx*panSpeed)).add(up.multiplyScalar(dy*panSpeed));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            }
          });
          dom.addEventListener('wheel', e=>{ e.preventDefault(); const s=Math.pow(1.1, Math.sign(e.deltaY)); spherical.radius=Math.max(0.001, spherical.radius*s);
            tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
          }, { passive:false });
          this.target = target; this.update = function(){};
        };
      })();
    }
  </script>
  <script>
    if (!window.THREE || !THREE.OrbitControls) {
      (function(){
        if (!window.THREE) return;
        THREE.OrbitControls = function(camera, dom) {
          const target = new THREE.Vector3();
          const spherical = new THREE.Spherical();
          const tmp = new THREE.Vector3();
          function sync(){ tmp.copy(camera.position).sub(target); spherical.setFromVector3(tmp); }
          sync();
          let dragging=false, panning=false, lastX=0, lastY=0;
          dom.addEventListener('contextmenu', e=>e.preventDefault());
          dom.addEventListener('mousedown', e=>{ if(e.button===0) dragging=true; else panning=true; lastX=e.clientX; lastY=e.clientY; });
          window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
          dom.addEventListener('mousemove', e=>{
            const dx=e.clientX-lastX, dy=e.clientY-lastY; lastX=e.clientX; lastY=e.clientY;
            if(dragging){
              spherical.theta -= dx*0.005; spherical.phi -= dy*0.005; spherical.phi=Math.max(0.001,Math.min(Math.PI-0.001,spherical.phi));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            } else if(panning){
              const panSpeed = spherical.radius*0.001; const forward=camera.getWorldDirection(new THREE.Vector3());
              const right=new THREE.Vector3().crossVectors(forward,camera.up).normalize(); const up=camera.up.clone().normalize();
              target.add(right.multiplyScalar(-dx*panSpeed)).add(up.multiplyScalar(dy*panSpeed));
              tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
            }
          });
          dom.addEventListener('wheel', e=>{ e.preventDefault(); const s=Math.pow(1.1, Math.sign(e.deltaY)); spherical.radius=Math.max(0.001, spherical.radius*s);
            tmp.setFromSpherical(spherical); camera.position.copy(target).add(tmp); camera.lookAt(target);
          }, { passive:false });
          this.target = target; this.update = function(){};
        };
      })();
    }
  </script>
  <script id=\"full-json\" type=\"application/json\">__FULL_JSON__</script>
  <script id=\"partial-method-json\" type=\"application/json\">__METHOD_JSON__</script>
  __GT_SCRIPT__
</head>
<body>
  <div id=\"toolbar\">
    <strong>Interactive Viewer:</strong>
    <span style=\"margin-left:12px;\">Left: Full (continuous)</span>
    <span style=\"margin-left:12px;\">Right: Partial (__PARTIAL_LABEL__)</span>
    __TOGGLE_BTN__
  </div>
  <div id=\"container\">
    <div class=\"panel\"><div class=\"label\">Full Mesh (continuous)</div><div id=\"left\"></div></div>
    <div class=\"panel\"><div class=\"label\">Partial Mesh (__PARTIAL_LABEL__)</div><div id=\"right\"></div></div>
  </div>
  <script>
    function getJSON(id) {
      const el = document.getElementById(id);
      if (!el) return null;
      try { return JSON.parse(el.textContent); } catch (e) { return null; }
    }

    function makeScene(dom) {
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xF9FAFB);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      dom.appendChild(renderer.domElement);
      const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      const light = new THREE.DirectionalLight(0xffffff, 0.9);
      light.position.set(1,1,1);
      scene.add(light);
      const amb = new THREE.AmbientLight(0xffffff, 0.3);
      scene.add(amb);
      function onResize() {
        const rect = (dom.parentElement || dom).getBoundingClientRect();
        renderer.setSize(rect.width, rect.height);
        camera.aspect = rect.width / rect.height;
        camera.updateProjectionMatrix();
      }
      window.addEventListener('resize', onResize);
      onResize();
      return { scene, renderer, camera, controls };
    }

    function buildMesh(data) {
      const verts = data.vertices;
      const faces = data.faces;
      const cols = data.colors;
      const geom = new THREE.BufferGeometry();
      const position = new Float32Array(verts.length * 3);
      const color = new Float32Array(verts.length * 3);
      for (let i=0;i<verts.length;i++) {
        position[3*i+0] = verts[i][0];
        position[3*i+1] = verts[i][1];
        position[3*i+2] = verts[i][2];
        color[3*i+0] = cols[i][0];
        color[3*i+1] = cols[i][1];
        color[3*i+2] = cols[i][2];
      }
      const index = new Uint32Array(faces.length * 3);
      for (let f=0; f<faces.length; f++) {
        index[3*f+0] = faces[f][0];
        index[3*f+1] = faces[f][1];
        index[3*f+2] = faces[f][2];
      }
      geom.setAttribute('position', new THREE.BufferAttribute(position, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(color, 3));
      geom.setIndex(new THREE.BufferAttribute(index, 1));
      geom.computeVertexNormals();
      const mat = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide });
      return new THREE.Mesh(geom, mat);
    }

    async function init() {
      const left = makeScene(document.getElementById('left'));
      const right = makeScene(document.getElementById('right'));

      const full = getJSON('full-json');
      const partialMethod = getJSON('partial-method-json');
      const partialGT = getJSON('partial-gt-json');

      const fullMesh = buildMesh(full);
      left.scene.add(fullMesh);
      const pmMesh = buildMesh(partialMethod);
      pmMesh.name = 'partial_method';
      right.scene.add(pmMesh);
      let pgMesh = null;
      if (partialGT) {
        pgMesh = buildMesh(partialGT);
        pgMesh.name = 'partial_gt';
        right.scene.add(pgMesh);
        pgMesh.visible = false;
      }

      function fitCamera(sceneObj, mesh) {
        const box = new THREE.Box3().setFromObject(mesh);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const dist = maxDim * 1.5;
        sceneObj.camera.position.copy(center.clone().add(new THREE.Vector3(dist, dist, dist)));
        sceneObj.controls.target.copy(center);
        sceneObj.controls.update();
      }

      fitCamera(left, fullMesh);
      fitCamera(right, pmMesh);

      function animate() {
        left.renderer.render(left.scene, left.camera);
        right.renderer.render(right.scene, right.camera);
        requestAnimationFrame(animate);
      }
      animate();

      if (pgMesh) {
        document.getElementById('toggleBtn').addEventListener('click', () => {
          const showGT = !pgMesh.visible;
          pgMesh.visible = showGT;
          pmMesh.visible = !showGT;
        });
      }
    }

    init();
  </script>
</body>
</html>
"""

    html = html.replace('__TOGGLE_BTN__', toggle_btn_html)
    html = html.replace('__PARTIAL_LABEL__', partial_label)
    html = html.replace('__GT_SCRIPT__', gt_script)
    html = html.replace('__FULL_JSON__', json.dumps(full_dict))
    html = html.replace('__METHOD_JSON__', json.dumps(method_dict))

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_interactive_view(full_mesh_path: str, partial_mesh_path: str, matches: np.ndarray, gt_matches: np.ndarray | None, out_dir: str, embed_assets: bool = True) -> str:
  os.makedirs(out_dir, exist_ok=True)
  # Ensure local three.js is bundled next to HTML so users can open without extra setup
  _ensure_three_js(out_dir)

  mesh_full = o3d.io.read_triangle_mesh(full_mesh_path)
  mesh_partial = o3d.io.read_triangle_mesh(partial_mesh_path)
  verts_M = np.asarray(mesh_full.vertices)
  faces_M = np.asarray(mesh_full.triangles)
  verts_N = np.asarray(mesh_partial.vertices)
  faces_N = np.asarray(mesh_partial.triangles)

  if not isinstance(matches, np.ndarray):
    matches = np.asarray(matches, dtype=int)
  colors_M = _hsv_colormap(verts_M.shape[0])
  colors_N_method = colors_M[matches]

  has_gt = gt_matches is not None and len(gt_matches) == verts_N.shape[0]
  colors_N_gt = None
  if has_gt:
    gt_idx = np.asarray(gt_matches, dtype=int)
    colors_N_gt = colors_M[gt_idx]

  html_path = os.path.join(out_dir, 'interactive_view.html')
  if embed_assets:
    full_dict = _mesh_to_json_dict(verts_M, faces_M, colors_M)
    method_dict = _mesh_to_json_dict(verts_N, faces_N, colors_N_method)
    gt_dict = _mesh_to_json_dict(verts_N, faces_N, colors_N_gt) if (has_gt and colors_N_gt is not None) else None
    _write_html_embedded(html_path, has_gt, full_dict, method_dict, gt_dict)
  else:
    # Write JSON assets alongside HTML
    _write_json(os.path.join(out_dir, 'full.json'), _mesh_to_json_dict(verts_M, faces_M, colors_M))
    _write_json(os.path.join(out_dir, 'partial_method.json'), _mesh_to_json_dict(verts_N, faces_N, colors_N_method))
    if has_gt and colors_N_gt is not None:
      _write_json(os.path.join(out_dir, 'partial_gt.json'), _mesh_to_json_dict(verts_N, faces_N, colors_N_gt))
    _write_html(html_path, has_gt)
  return html_path
