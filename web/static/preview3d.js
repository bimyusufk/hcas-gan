const viewport = document.getElementById('preview3dViewport');
const repeatXEl = document.getElementById('repeatX');
const repeatYEl = document.getElementById('repeatY');
const rotationEl = document.getElementById('textureRotation');
const statsEl = document.getElementById('preview3dStats');

function setPreviewStats() {
  if (!statsEl) return;
  const rx = Number(repeatXEl?.value || 4).toFixed(1);
  const ry = Number(repeatYEl?.value || 4).toFixed(1);
  const rot = Number(rotationEl?.value || 0).toFixed(0);
  statsEl.textContent = `RepeatX: ${rx} · RepeatY: ${ry} · Rotasi: ${rot}°`;
}
if (!viewport) {
  throw new Error('Missing #preview3dViewport element');
}

async function loadThreeModules() {
  const sources = [
    {
      three: 'https://esm.sh/three@0.164.1',
      controls: 'https://esm.sh/three@0.164.1/examples/jsm/controls/OrbitControls.js',
      gltf: 'https://esm.sh/three@0.164.1/examples/jsm/loaders/GLTFLoader.js',
    },
    {
      three: 'https://unpkg.com/three@0.164.1/build/three.module.js?module',
      controls: 'https://unpkg.com/three@0.164.1/examples/jsm/controls/OrbitControls.js?module',
      gltf: 'https://unpkg.com/three@0.164.1/examples/jsm/loaders/GLTFLoader.js?module',
    },
  ];

  const errors = [];
  for (const src of sources) {
    try {
      const threeMod = await import(src.three);
      const controlsMod = await import(src.controls);
      const gltfMod = await import(src.gltf);
      return {
        THREE: threeMod,
        OrbitControls: controlsMod.OrbitControls,
        GLTFLoader: gltfMod.GLTFLoader,
      };
    } catch (err) {
      errors.push(String(err?.message || err));
    }
  }

  throw new Error(`Tidak bisa memuat modul 3D dari CDN. Detail: ${errors.join(' | ')}`);
}

function fitCameraToObject(THREE, camera, controls, object3D, offset = 1.25) {
  const box = new THREE.Box3().setFromObject(object3D);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxSize = Math.max(size.x, size.y, size.z, 0.001);
  const fitHeightDistance = maxSize / (2 * Math.tan((Math.PI * camera.fov) / 360));
  const fitWidthDistance = fitHeightDistance / camera.aspect;
  const distance = offset * Math.max(fitHeightDistance, fitWidthDistance);

  camera.near = Math.max(0.01, distance / 100);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  camera.position.set(center.x, center.y + maxSize * 0.2, center.z + distance);
  controls.target.copy(center);
  controls.update();
}

(async () => {
  let renderer;
  let scene;
  let camera;
  let controls;
  let textureLoader;
  let THREE;
  let GLTFLoader;

  let shirtMeshMaterials = [];
  let activeTexture = null;
  let fallbackTexture = null;

  const setError = (message) => {
    if (statsEl) {
      statsEl.textContent = `Gagal load preview 3D: ${message}`;
    }
  };

  const applyTextureParams = (texture) => {
    if (!texture || !THREE) return;
    const rx = Number(repeatXEl?.value || 4);
    const ry = Number(repeatYEl?.value || 4);
    const deg = Number(rotationEl?.value || 0);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.center.set(0.5, 0.5);
    texture.repeat.set(rx, ry);
    texture.rotation = THREE.MathUtils.degToRad(deg);
    texture.needsUpdate = true;
    setPreviewStats();
  };

  const applyTextureToShirt = (texture) => {
    if (!texture) return;
    applyTextureParams(texture);
    shirtMeshMaterials.forEach((material) => {
      material.map = texture;
      material.needsUpdate = true;
    });
  };

  const resizeRenderer = () => {
    if (!renderer || !camera) return;
    const w = Math.max(320, viewport.clientWidth || 320);
    const h = Math.max(360, viewport.clientHeight || 460);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };

  const animate = () => {
    if (!renderer || !scene || !camera || !controls) return;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };

  try {
    const loaded = await loadThreeModules();
    THREE = loaded.THREE;
    const OrbitControls = loaded.OrbitControls;
    GLTFLoader = loaded.GLTFLoader;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    viewport.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.background = null;

    camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    camera.position.set(0.0, 1.2, 3.0);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 0.2;
    controls.maxDistance = 20;

    scene.add(new THREE.AmbientLight(0xffffff, 1.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
    keyLight.position.set(2.5, 4.0, 2.0);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xa9c7ff, 0.55);
    rimLight.position.set(-2.0, 2.0, -3.0);
    scene.add(rimLight);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(4.0, 64),
      new THREE.MeshStandardMaterial({ color: 0x111723, roughness: 0.95, metalness: 0.0 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.02;
    scene.add(floor);

    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    const cx = c.getContext('2d');
    cx.fillStyle = '#3f4f5d';
    cx.fillRect(0, 0, 128, 128);
    cx.fillStyle = '#5f7386';
    cx.fillRect(0, 0, 64, 64);
    cx.fillRect(64, 64, 64, 64);
    fallbackTexture = new THREE.CanvasTexture(c);
    fallbackTexture.colorSpace = THREE.SRGBColorSpace;
    fallbackTexture.wrapS = THREE.RepeatWrapping;
    fallbackTexture.wrapT = THREE.RepeatWrapping;
    fallbackTexture.repeat.set(4, 4);

    textureLoader = new THREE.TextureLoader();

    const modelUrl = window.HCAS_MODEL_GLB_URL || '/model/long_sleeve_t-_shirt.glb';
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(modelUrl);

    const shirtRoot = gltf.scene;
    shirtRoot.rotation.y = Math.PI;

    const bbox = new THREE.Box3().setFromObject(shirtRoot);
    const size = bbox.getSize(new THREE.Vector3());
    const maxAxis = Math.max(size.x, size.y, size.z, 1e-6);
    const targetHeight = 2.0;
    const scale = targetHeight / maxAxis;
    shirtRoot.scale.setScalar(scale);

    shirtMeshMaterials = [];
    shirtRoot.traverse((obj) => {
      if (!obj.isMesh) return;
      const m = obj.material;
      if (Array.isArray(m)) {
        m.forEach((mat) => {
          if (mat && !shirtMeshMaterials.includes(mat)) {
            mat.roughness = 0.85;
            mat.metalness = 0.05;
            shirtMeshMaterials.push(mat);
          }
        });
      } else if (m && !shirtMeshMaterials.includes(m)) {
        m.roughness = 0.85;
        m.metalness = 0.05;
        shirtMeshMaterials.push(m);
      }
    });

    scene.add(shirtRoot);
    resizeRenderer();
    fitCameraToObject(THREE, camera, controls, shirtRoot, 1.2);

    applyTextureToShirt(fallbackTexture);

    window.update3DTexture = (src) => {
      if (!src || shirtMeshMaterials.length === 0) return;
      textureLoader.load(
        src,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          activeTexture = texture;
          applyTextureToShirt(activeTexture);
        },
        undefined,
        () => {
          applyTextureToShirt(fallbackTexture);
        }
      );
    };

    [repeatXEl, repeatYEl, rotationEl].forEach((el) => {
      if (!el) return;
      el.addEventListener('input', () => {
        if (activeTexture) {
          applyTextureToShirt(activeTexture);
        } else {
          applyTextureToShirt(fallbackTexture);
        }
      });
    });

    window.addEventListener('resize', resizeRenderer);

    const pending = window.__latestPatternTexture;
    if (pending) {
      window.update3DTexture(pending);
    }
    setPreviewStats();
    animate();
  } catch (err) {
    setError(err?.message || String(err));
  }
})();
