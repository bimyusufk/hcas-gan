const HCAS3D = (() => {
  const handles = [];

  let modulesPromise = null;
  async function loadThreeModules() {
    if (modulesPromise) {
      return modulesPromise;
    }

    modulesPromise = (async () => {
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

      throw new Error(`Tidak bisa memuat modul 3D: ${errors.join(' | ')}`);
    })();

    return modulesPromise;
  }

  function fitCameraToObject(THREE, camera, controls, object3D, offset = 1.2) {
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

  async function createPreview(containerEl, textureSrc, opts = {}) {
    if (!containerEl) {
      return null;
    }

    const modelUrl = opts.modelUrl || '/model/long_sleeve_t-_shirt.glb';
    const { THREE, OrbitControls, GLTFLoader } = await loadThreeModules();

    containerEl.innerHTML = '';

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    containerEl.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = null;

    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    camera.position.set(0.0, 1.2, 3.0);

    const controls = new OrbitControls(camera, renderer.domElement);
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

    const loader = new GLTFLoader();
    const textureLoader = new THREE.TextureLoader();
    const gltf = await loader.loadAsync(modelUrl);
    const shirtRoot = gltf.scene;
    shirtRoot.rotation.y = Math.PI;

    const bbox = new THREE.Box3().setFromObject(shirtRoot);
    const size = bbox.getSize(new THREE.Vector3());
    const maxAxis = Math.max(size.x, size.y, size.z, 1e-6);
    const scale = 2.0 / maxAxis;
    shirtRoot.scale.setScalar(scale);

    const shirtMaterials = [];
    shirtRoot.traverse((obj) => {
      if (!obj.isMesh) return;
      const material = obj.material;
      if (Array.isArray(material)) {
        material.forEach((mat) => {
          if (mat && !shirtMaterials.includes(mat)) {
            mat.roughness = 0.85;
            mat.metalness = 0.05;
            shirtMaterials.push(mat);
          }
        });
      } else if (material && !shirtMaterials.includes(material)) {
        material.roughness = 0.85;
        material.metalness = 0.05;
        shirtMaterials.push(material);
      }
    });

    scene.add(shirtRoot);

    // Texture configuration state
    const textureConfig = {
      repeatX: 1,
      repeatY: 1,
      offsetX: 0,
      offsetY: 0,
      rotation: 0,
      currentTexture: null,
    };

    const REPEAT_MAX = 3.0;
    const clampRepeatValue = (value) => {
      const numeric = Number.isFinite(value) ? value : 0;
      return Math.min(REPEAT_MAX, Math.max(0, numeric));
    };

    const getEffectiveRepeat = (value) => {
      const clamped = clampRepeatValue(value);
      return clamped <= 0 ? 1 : clamped;
    };

    const applyTextureTransform = (texture) => {
      texture.wrapS = textureConfig.repeatX <= 0 ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
      texture.wrapT = textureConfig.repeatY <= 0 ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
      texture.center.set(0.5, 0.5);
      texture.repeat.set(getEffectiveRepeat(textureConfig.repeatX), getEffectiveRepeat(textureConfig.repeatY));
      texture.offset.set(textureConfig.offsetX, textureConfig.offsetY);
      texture.rotation = (textureConfig.rotation * Math.PI) / 180;
      texture.needsUpdate = true;
    };

    const applyTexture = (texture) => {
      applyTextureTransform(texture);
      textureConfig.currentTexture = texture;
      shirtMaterials.forEach((material) => {
        material.map = texture;
        material.needsUpdate = true;
      });
    };

    const updateTextureUniforms = () => {
      if (textureConfig.currentTexture) {
        applyTextureTransform(textureConfig.currentTexture);
      }
    };

    const fallbackCanvas = document.createElement('canvas');
    fallbackCanvas.width = 128;
    fallbackCanvas.height = 128;
    const fallbackCtx = fallbackCanvas.getContext('2d');
    fallbackCtx.fillStyle = '#3f4f5d';
    fallbackCtx.fillRect(0, 0, 128, 128);
    fallbackCtx.fillStyle = '#5f7386';
    fallbackCtx.fillRect(0, 0, 64, 64);
    fallbackCtx.fillRect(64, 64, 64, 64);
    const fallbackTexture = new THREE.CanvasTexture(fallbackCanvas);
    fallbackTexture.colorSpace = THREE.SRGBColorSpace;
    applyTexture(fallbackTexture);

    if (textureSrc) {
      textureLoader.load(
        textureSrc,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          applyTexture(texture);
        },
        undefined,
        () => {
          applyTexture(fallbackTexture);
        }
      );
    }

    const resize = () => {
      const w = Math.max(220, containerEl.clientWidth || 220);
      const h = Math.max(220, containerEl.clientHeight || 220);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      fitCameraToObject(THREE, camera, controls, shirtRoot, 1.2);
    };

    let active = true;
    const renderLoop = () => {
      if (!active) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(renderLoop);
    };

    resize();
    renderLoop();

    const onResize = () => resize();
    window.addEventListener('resize', onResize);

    // Create UI controls wrapper
    const controls3dUI = document.createElement('div');
    controls3dUI.className = 'preview3d-controls';
    
    const createSlider = (label, min, max, value, step = 0.1, onChange) => {
      const container = document.createElement('div');
      container.className = 'preview3d-control-item';
      
      const labelEl = document.createElement('label');
      labelEl.className = 'preview3d-control-label';
      labelEl.textContent = label;
      
      const input = document.createElement('input');
      input.type = 'range';
      input.min = min;
      input.max = max;
      input.step = step;
      input.value = value;
      input.className = 'preview3d-control-slider';
      
      const valueDisplay = document.createElement('span');
      valueDisplay.className = 'preview3d-control-value';
      valueDisplay.textContent = value;
      
      input.addEventListener('input', (e) => {
        const newValue = parseFloat(e.target.value);
        valueDisplay.textContent = newValue.toFixed(step < 1 ? 2 : 0);
        onChange(newValue);
      });
      
      container.appendChild(labelEl);
      container.appendChild(input);
      container.appendChild(valueDisplay);
      return container;
    };

    controls3dUI.appendChild(createSlider('Repeat X', 0, REPEAT_MAX, 1, 0.1, (v) => {
      textureConfig.repeatX = clampRepeatValue(v);
      updateTextureUniforms();
    }));
    controls3dUI.appendChild(createSlider('Repeat Y', 0, REPEAT_MAX, 1, 0.1, (v) => {
      textureConfig.repeatY = clampRepeatValue(v);
      updateTextureUniforms();
    }));
    controls3dUI.appendChild(createSlider('Offset X', -1, 1, 0, 0.01, (v) => {
      textureConfig.offsetX = v;
      updateTextureUniforms();
    }));
    controls3dUI.appendChild(createSlider('Offset Y', -1, 1, 0, 0.01, (v) => {
      textureConfig.offsetY = v;
      updateTextureUniforms();
    }));
    controls3dUI.appendChild(createSlider('Rotation', 0, 360, 0, 1, (v) => {
      textureConfig.rotation = v;
      updateTextureUniforms();
    }));

    // Fullscreen modal
    const createFullscreenModal = () => {
      const modal = document.createElement('div');
      modal.className = 'preview3d-modal';
      modal.style.display = 'none';
      
      const backdrop = document.createElement('div');
      backdrop.className = 'preview3d-modal-backdrop';
      backdrop.addEventListener('click', () => {
        modal.style.display = 'none';
      });
      
      const content = document.createElement('div');
      content.className = 'preview3d-modal-content';
      content.addEventListener('click', (e) => e.stopPropagation());
      
      const closeBtn = document.createElement('button');
      closeBtn.className = 'preview3d-modal-close';
      closeBtn.textContent = '✕';
      closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
      });
      
      const fullscreenViewport = document.createElement('div');
      fullscreenViewport.className = 'preview3d-fullscreen-viewport';
      
      const fullscreenControls = document.createElement('div');
      fullscreenControls.className = 'preview3d-fullscreen-controls';
      
      fullscreenControls.appendChild(createSlider('Repeat X', 0, REPEAT_MAX, 1, 0.5, (v) => {
        textureConfig.repeatX = clampRepeatValue(v);
        updateTextureUniforms();
      }));
      fullscreenControls.appendChild(createSlider('Repeat Y', 0, REPEAT_MAX, 1, 0.5, (v) => {
        textureConfig.repeatY = clampRepeatValue(v);
        updateTextureUniforms();
      }));
      fullscreenControls.appendChild(createSlider('Offset X', -1, 1, 0, 0.01, (v) => {
        textureConfig.offsetX = v;
        updateTextureUniforms();
      }));
      fullscreenControls.appendChild(createSlider('Offset Y', -1, 1, 0, 0.01, (v) => {
        textureConfig.offsetY = v;
        updateTextureUniforms();
      }));
      fullscreenControls.appendChild(createSlider('Rotation', 0, 360, 0, 1, (v) => {
        textureConfig.rotation = v;
        updateTextureUniforms();
      }));
      
      content.appendChild(closeBtn);
      content.appendChild(fullscreenViewport);
      content.appendChild(fullscreenControls);
      modal.appendChild(backdrop);
      modal.appendChild(content);
      document.body.appendChild(modal);
      
      return { modal, fullscreenViewport };
    };

    const { modal: fullscreenModal, fullscreenViewport } = createFullscreenModal();

    // Add click handler to canvas for fullscreen
    const onCanvasClick = (e) => {
      e.preventDefault();
      fullscreenModal.style.display = 'flex';
      
      // Remove old canvas and create new one in fullscreen
      while (fullscreenViewport.firstChild) {
        fullscreenViewport.removeChild(fullscreenViewport.firstChild);
      }

      const fullscreenRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      fullscreenRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      fullscreenRenderer.outputColorSpace = THREE.SRGBColorSpace;
      fullscreenRenderer.toneMapping = THREE.ACESFilmicToneMapping;
      fullscreenRenderer.toneMappingExposure = 1.0;
      fullscreenViewport.appendChild(fullscreenRenderer.domElement);

      const fullscreenScene = new THREE.Scene();
      fullscreenScene.background = null;

      const fullscreenCamera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
      fullscreenCamera.position.set(0.0, 1.2, 3.0);

      const fullscreenControls = new OrbitControls(fullscreenCamera, fullscreenRenderer.domElement);
      fullscreenControls.enableDamping = true;
      fullscreenControls.minDistance = 0.2;
      fullscreenControls.maxDistance = 20;

      fullscreenScene.add(new THREE.AmbientLight(0xffffff, 1.15));
      const fullscreenKeyLight = new THREE.DirectionalLight(0xffffff, 1.1);
      fullscreenKeyLight.position.set(2.5, 4.0, 2.0);
      fullscreenScene.add(fullscreenKeyLight);
      const fullscreenRimLight = new THREE.DirectionalLight(0xa9c7ff, 0.55);
      fullscreenRimLight.position.set(-2.0, 2.0, -3.0);
      fullscreenScene.add(fullscreenRimLight);

      const fullscreenFloor = new THREE.Mesh(
        new THREE.CircleGeometry(4.0, 64),
        new THREE.MeshStandardMaterial({ color: 0x111723, roughness: 0.95, metalness: 0.0 })
      );
      fullscreenFloor.rotation.x = -Math.PI / 2;
      fullscreenFloor.position.y = -0.02;
      fullscreenScene.add(fullscreenFloor);

      const clonedShirtRoot = shirtRoot.clone();
      fullscreenScene.add(clonedShirtRoot);

      const fullscreenShirtMaterials = [];
      clonedShirtRoot.traverse((obj) => {
        if (!obj.isMesh) return;
        const material = obj.material;
        if (Array.isArray(material)) {
          material.forEach((mat) => {
            if (mat && !fullscreenShirtMaterials.includes(mat)) {
              mat.map = textureConfig.currentTexture;
              mat.needsUpdate = true;
              fullscreenShirtMaterials.push(mat);
            }
          });
        } else if (material && !fullscreenShirtMaterials.includes(material)) {
          material.map = textureConfig.currentTexture;
          material.needsUpdate = true;
          fullscreenShirtMaterials.push(material);
        }
      });

      const fullscreenResize = () => {
        const w = fullscreenViewport.clientWidth || window.innerWidth;
        const h = fullscreenViewport.clientHeight || window.innerHeight;
        fullscreenRenderer.setSize(w, h, false);
        fullscreenCamera.aspect = w / h;
        fullscreenCamera.updateProjectionMatrix();
        fitCameraToObject(THREE, fullscreenCamera, fullscreenControls, clonedShirtRoot, 1.2);
      };

      let fullscreenActive = true;
      const fullscreenRenderLoop = () => {
        if (!fullscreenActive) return;
        fullscreenControls.update();
        fullscreenRenderer.render(fullscreenScene, fullscreenCamera);
        requestAnimationFrame(fullscreenRenderLoop);
      };

      fullscreenResize();
      fullscreenRenderLoop();

      const onFullscreenResize = () => fullscreenResize();
      window.addEventListener('resize', onFullscreenResize);

      const closeFullscreen = () => {
        fullscreenActive = false;
        window.removeEventListener('resize', onFullscreenResize);
        try {
          fullscreenRenderer.dispose();
        } catch (_) {
          // no-op
        }
      };

      const backdrop = fullscreenModal.querySelector('.preview3d-modal-backdrop');
      const handleBackdropClick = (e) => {
        if (e.target === backdrop) {
          closeFullscreen();
          fullscreenModal.style.display = 'none';
        }
      };

      backdrop.addEventListener('click', handleBackdropClick);

      // Update close button event
      const closeBtn = fullscreenModal.querySelector('.preview3d-modal-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          closeFullscreen();
          fullscreenModal.style.display = 'none';
        });
      }
    };

    renderer.domElement.style.cursor = 'pointer';
    renderer.domElement.addEventListener('click', onCanvasClick);

    const handle = {
      dispose: () => {
        active = false;
        window.removeEventListener('resize', onResize);
        renderer.domElement.removeEventListener('click', onCanvasClick);
        try {
          renderer.dispose();
        } catch (_) {
          // no-op
        }
        if (renderer.domElement?.parentElement) {
          renderer.domElement.parentElement.removeChild(renderer.domElement);
        }
      },
      getControlsUI: () => controls3dUI,
    };

    handles.push(handle);
    return handle;
  }

  function disposeAll() {
    while (handles.length > 0) {
      const handle = handles.pop();
      try {
        handle.dispose();
      } catch (_) {
        // no-op
      }
    }
  }

  return {
    createPreview,
    disposeAll,
  };
})();

window.HCAS3D = HCAS3D;