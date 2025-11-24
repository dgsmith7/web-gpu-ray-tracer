// WebGPU progressive path tracer host
//
// Notes:
// - The Camera UBO layout, sphere packing (16 floats/sphere), and seed buffer
//   length must match the WGSL definitions exactly (padding matters).

const canvas = document.getElementById("canvas");

// - Progressive path tracer: dispatches one or more stochastic samples per
//   frame and accumulates linear radiance into `rgba32float` accumulation
//   textures (ping-pong). The host controls `SAMPLES_PER_FRAME` and
//   `TARGET_SAMPLES` (total samples-per-pixel) and stops/logs timing when
//   the target is reached.
// - Increasing `SAMPLES_PER_FRAME` speeds convergence per frame but increases
//   GPU work per dispatch. Increasing `MAX_DEPTH` (in WGSL) increases per-ray
//   work.
//   Recommend leaving `SAMPLES_PER_FRAME` at 1 for smooth progressive rendering
//   and setting `TARGET_SAMPLES` to desired quality.
//  (SAMPLES_PER_FRAME is a "shaderism" that isn't used in the book).
// Must match WGSL SAMPLES_PER_FRAME (1 for progressive rendering)
const SAMPLES_PER_FRAME = 1;
// Set how many samples-per-pixel constitute a "complete" render.
const TARGET_SAMPLES = 500; // "samples per pixel" in book terms

// - Per-pixel PRNG: a `u32` state per pixel is seeded on the CPU with
//   `crypto.getRandomValues` (chunked) and advanced in WGSL via xorshift32.
// createSeedBuffer: initialize per-pixel PRNG state.
// Each pixel stores a u32 state that WGSL advances (xorshift32) so each
// progressive sample uses a fresh random sequence. We chunk `getRandomValues`
// to avoid browser quota limits for large buffers.  Crypto is non-deterministic,
// so each run will produce different renders.  Math.random() is weaker and used as a fallback.
async function createSeedBuffer(device, width, height) {
  const pixelCount = width * height;
  const seedArray = new Uint32Array(pixelCount);

  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    const MAX_BYTES = 65536;
    const BYTES_PER_ELEM = Uint32Array.BYTES_PER_ELEMENT;
    const maxElemsPerCall = Math.floor(MAX_BYTES / BYTES_PER_ELEM); // 16384
    let offset = 0;
    while (offset < pixelCount) {
      const chunk = Math.min(maxElemsPerCall, pixelCount - offset);
      const view = new Uint32Array(
        seedArray.buffer,
        offset * BYTES_PER_ELEM,
        chunk
      );
      crypto.getRandomValues(view);
      offset += chunk;
    }
  } else {
    for (let i = 0; i < pixelCount; ++i)
      seedArray[i] = (Math.random() * 0xffffffff) >>> 0;
    console.warn(
      "crypto.getRandomValues unavailable — seeded with Math.random()"
    );
  }

  // Avoid 'zero' seeds: xorshift32 (method used in shader) will remain zero if initial state == 0.
  // Replace any zero with a fixed nonzero constant (e.g. golden ratio constant).
  // This is to avoid a zero seed from producing zero forever in the shader.
  const NONZERO_FALLBACK = 0x9e3779b9 >>> 0;
  for (let i = 0; i < pixelCount; ++i) {
    if (seedArray[i] === 0) seedArray[i] = NONZERO_FALLBACK;
  }

  const seedBuffer = device.createBuffer({
    size: seedArray.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(
    seedBuffer,
    0,
    seedArray.buffer,
    seedArray.byteOffset,
    seedArray.byteLength
  );
  return seedBuffer;
}
//////////////////////////////////////////////////

// small math helpers - simple vector ops needed for set-up of camera and scene
function vec3(x, y, z) {
  return { x, y, z };
}
function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}
function length(v) {
  return Math.hypot(v.x, v.y, v.z);
}
function mulScalar(v, s) {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}
function add(a, b) {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}
function cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}
function normalize(v) {
  const L = Math.hypot(v.x, v.y, v.z) || 1.0;
  return { x: v.x / L, y: v.y / L, z: v.z / L };
}
//////////////////////////////////////////////////

// - Camera: CPU builds camera vectors (`origin`, `lower_left`, `horizontal`,
//   `vertical`) and packs them into an 80-byte Camera UBO that the shader
//   reads. To support depth-of-field, the CPU writes a `lensRadius` into the
//   UBO padding slot (`_pad3`) so the shader can sample a lens disk.
// buildCamera: matches the book's simple pinhole camera.
// The returned vectors are packed into a Camera UBO (see below) which the
// shader uses to construct primary rays per pixel.
function buildCamera(
  lookfrom,
  lookat,
  vup,
  vfov_deg,
  aspect,
  focus_dist = 1.0
) {
  const theta = (vfov_deg * Math.PI) / 180.0;
  const h = Math.tan(theta / 2.0);
  const viewport_height = 2.0 * h;
  const viewport_width = aspect * viewport_height;

  const w = normalize({
    x: lookfrom.x - lookat.x,
    y: lookfrom.y - lookat.y,
    z: lookfrom.z - lookat.z,
  });
  const u = normalize(cross(vup, w));
  const v = cross(w, u);

  const origin = lookfrom;
  const horizontal = mulScalar(u, viewport_width * focus_dist);
  const vertical = mulScalar(v, viewport_height * focus_dist);
  const lower_left = sub(
    sub(add(origin, mulScalar(horizontal, -0.5)), mulScalar(vertical, 0.5)),
    mulScalar(w, focus_dist)
  );
  return { origin, lower_left, horizontal, vertical };
}
//////////////////////////////////////////////////

// Build scene (book-like) and pack into Float32Array for WGSL storage buffer.
// buildSceneBuffer: creates the book-like final scene and packs each sphere
// into 16 floats (4 vec4s) so the WGSL `Sphere` storage struct can read them
// directly. Layout per-sphere: (cx,cy,cz,r), (albedo.r,g,b,kind), (fuzz,ref,emit.r,emit.g), (emit.b, pad...)
function buildSceneBuffer() {
  const spheres = [];
  spheres.push({
    center: { x: 0, y: -1000, z: 0 },
    radius: 1000,
    albedo: { r: 0.5, g: 0.5, b: 0.5 },
    kind: 0,
    fuzz: 0,
    ref_idx: 1.0,
    emission: { r: 0, g: 0, b: 0 },
  });

  for (let a = -11; a < 11; ++a) {
    for (let b = -11; b < 11; ++b) {
      const chooseMat = Math.random();
      const center = {
        x: a + 0.9 * Math.random(),
        y: 0.2,
        z: b + 0.9 * Math.random(),
      };
      if (length(sub(center, { x: 4, y: 0.2, z: 0 })) > 0.9) {
        if (chooseMat < 0.8) {
          const albedo = {
            r: Math.random() * Math.random(),
            g: Math.random() * Math.random(),
            b: Math.random() * Math.random(),
          };
          spheres.push({
            center,
            radius: 0.2,
            albedo,
            kind: 0,
            fuzz: 0.0,
            ref_idx: 1.0,
            emission: { r: 0, g: 0, b: 0 },
          });
        } else if (chooseMat < 0.95) {
          const albedo = {
            r: 0.5 + 0.5 * Math.random(),
            g: 0.5 + 0.5 * Math.random(),
            b: 0.5 + 0.5 * Math.random(),
          };
          const fuzz = 0.5 * Math.random();
          spheres.push({
            center,
            radius: 0.2,
            albedo,
            kind: 2,
            fuzz,
            ref_idx: 1.0,
            emission: { r: 0, g: 0, b: 0 },
          });
        } else {
          spheres.push({
            center,
            radius: 0.2,
            albedo: { r: 1, g: 1, b: 1 },
            kind: 3,
            fuzz: 0.0,
            ref_idx: 1.5,
            emission: { r: 0, g: 0, b: 0 },
          });
        }
      }
    }
  }

  spheres.push({
    center: { x: 0, y: 1, z: 0 },
    radius: 1.0,
    albedo: { r: 1, g: 1, b: 1 },
    kind: 3,
    fuzz: 0.0,
    ref_idx: 1.5,
    emission: { r: 0, g: 0, b: 0 },
  });
  spheres.push({
    center: { x: -4, y: 1, z: 0 },
    radius: 1.0,
    albedo: { r: 0.4, g: 0.2, b: 0.1 },
    kind: 0,
    fuzz: 0.0,
    ref_idx: 1.0,
    emission: { r: 0, g: 0, b: 0 },
  });
  spheres.push({
    center: { x: 4, y: 1, z: 0 },
    radius: 1.0,
    albedo: { r: 0.7, g: 0.6, b: 0.5 },
    kind: 2,
    fuzz: 0.0,
    ref_idx: 1.0,
    emission: { r: 0, g: 0, b: 0 },
  });

  const count = spheres.length;
  const floatCount = count * 16;
  const arr = new Float32Array(floatCount);
  let off = 0;
  for (let s of spheres) {
    arr[off++] = s.center.x;
    arr[off++] = s.center.y;
    arr[off++] = s.center.z;
    arr[off++] = s.radius;
    arr[off++] = s.albedo.r;
    arr[off++] = s.albedo.g;
    arr[off++] = s.albedo.b;
    arr[off++] = s.kind;
    arr[off++] = s.fuzz;
    arr[off++] = s.ref_idx;
    arr[off++] = s.emission.r;
    arr[off++] = s.emission.g;
    arr[off++] = s.emission.b;
    arr[off++] = 0.0;
    arr[off++] = 0.0;
    arr[off++] = 0.0;
  }
  return { array: arr, count };
}
//////////////////////////////////////////////////

async function init() {
  // WebGPU setup
  if (!("gpu" in navigator)) {
    alert("WebGPU not supported");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    alert("No adapter");
    return;
  }
  const device = await adapter.requestDevice();

  device.lost.then((info) => console.error("GPU device lost:", info));
  device.onuncapturederror = (e) => {
    console.error("Uncaptured GPU error:", e.error?.message ?? e);
    console.error(e.error);
  };

  const context = canvas.getContext("webgpu");
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "opaque",
  });

  // - Canvas sizing / DPR: the host optionally sets the canvas physical size
  //   (using `TARGET_WIDTH`/`TARGET_HEIGHT` scaled by `devicePixelRatio`) before
  //   creating textures and computing dispatch sizes so the camera, dispatch,
  //   and textures match a specific render resolution.
  // --- set logical image size (before reading canvas.width/height) ---
  // Change TARGET_WIDTH / TARGET_HEIGHT to the desired pixel dimensions.
  // Keeps CSS size for layout while rendering at device pixel ratio (DPR).
  const TARGET_WIDTH = 1280; // desired CSS width in pixels
  const TARGET_HEIGHT = 720; // desired CSS height in pixels
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(TARGET_WIDTH * dpr);
  canvas.height = Math.floor(TARGET_HEIGHT * dpr);
  canvas.style.width = `${TARGET_WIDTH}px`;
  canvas.style.height = `${TARGET_HEIGHT}px`;
  // ---------------------------------------------------------------
  // Read actual pixel dimensions from canvas for use in camera, textures, dispatch:
  const width = canvas.width;
  const height = canvas.height;
  const aspect = width / height;

  // Create a texture for display output (rgba8unorm)
  const displayTex = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC,
  });

  // Accumulation textures store linear sums of radiance with high precision.
  // Use `rgba32float` so we don't lose precision when summing many samples.
  // We ping-pong between two accumulation textures (accumA/accumB) each frame.
  // One is for read and one is for write, then we swap them.
  // Simultaneous read and write to one texture is not allowed.
  const accumFormat = "rgba32float";
  const accumDesc = {
    size: [width, height, 1],
    format: accumFormat,
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_SRC,
  };
  let accumA = device.createTexture(accumDesc);
  let accumB = device.createTexture(accumDesc);
  let accumSrc = accumA;
  let accumDst = accumB;

  // Build camera and pack into Camera Uniform Buffer Object (UBO).
  const lookfrom = vec3(13, 2, 3);
  const lookat = vec3(0, 0, 0);
  const vup = vec3(0, 1, 0);
  const vfov = 20.0;
  const focus_dist = 10.0;
  const cam = buildCamera(lookfrom, lookat, vup, vfov, aspect, focus_dist);
  // Depth of field: set aperture (0 => pinhole). lensRadius = aperture / 2.
  // Book uses "defocus_angle" terminology, which is actually aperture diameter.
  // - Depth of Field (DoF): the host exposes `aperture` -> `lensRadius` and
  //   the WGSL shader offsets ray origins on a sampled lens disk to implement
  //   defocus per the book (sample aperture / focal point logic).
  const aperture = 0.1; // tweak for stronger/weaker DOF
  const lensRadius = aperture * 0.5;

  // Camera UBO packing: MUST match `CameraUBO` layout in WGSL (vec3 + padding ordering)
  // We use 80 bytes: 4 vec3 fields (with padding) + two u32 resolution fields.
  // WGSL (and WebGPU buffer layout rules) require 16‑byte alignment for vec3/vec4-like data
  const cameraBufferSize = 80;
  const cameraArray = new ArrayBuffer(cameraBufferSize);
  const dv = new DataView(cameraArray);
  let off = 0;
  dv.setFloat32(off, cam.origin.x, true);
  off += 4;
  dv.setFloat32(off, cam.origin.y, true);
  off += 4;
  dv.setFloat32(off, cam.origin.z, true);
  off += 4;
  dv.setFloat32(off, 0.0, true);
  off += 4;

  dv.setFloat32(off, cam.lower_left.x, true);
  off += 4;
  dv.setFloat32(off, cam.lower_left.y, true);
  off += 4;
  dv.setFloat32(off, cam.lower_left.z, true);
  off += 4;
  dv.setFloat32(off, 0.0, true);
  off += 4;

  dv.setFloat32(off, cam.horizontal.x, true);
  off += 4;
  dv.setFloat32(off, cam.horizontal.y, true);
  off += 4;
  dv.setFloat32(off, cam.horizontal.z, true);
  off += 4;
  dv.setFloat32(off, 0.0, true);
  off += 4;

  dv.setFloat32(off, cam.vertical.x, true);
  off += 4;
  dv.setFloat32(off, cam.vertical.y, true);
  off += 4;
  dv.setFloat32(off, cam.vertical.z, true);
  off += 4;
  // store lens radius into the padding slot (_pad3) so shader can sample the lens
  dv.setFloat32(off, lensRadius, true);
  off += 4;

  dv.setUint32(off, width, true);
  off += 4;
  dv.setUint32(off, height, true);
  off += 4;
  dv.setUint32(off, 0, true);
  off += 4;
  dv.setUint32(off, 0, true);
  off += 4;

  const uniformBuffer = device.createBuffer({
    size: cameraBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, cameraArray);

  // Sample count UBO: holds the current number of accumulated samples.
  // The compute shader reads this to compute the running average.
  const sampleCountBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  let sampleCount = 0;
  device.queue.writeBuffer(
    sampleCountBuffer,
    0,
    new Uint32Array([sampleCount, 0, 0, 0])
  );

  // Timing / finish control:
  // renderStartTime measured when the first compute submission occurs.
  // finished flag prevents multiple logs and stops the loop once TARGET_SAMPLES reached.
  let renderStartTime = null;
  let renderStartISO = "";
  let finished = false;
  let frameIndex = 0;

  const seedBuffer = await createSeedBuffer(device, width, height);

  const scene = buildSceneBuffer();
  const sphereFloatArray = scene.array;
  const sphereCount = scene.count;
  // Holds the scene’s sphere data so the compute shader can read it for intersection tests.
  const sphereBuffer = device.createBuffer({
    size: sphereFloatArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sphereBuffer, 0, sphereFloatArray);
  // The shader reads sceneParams.sphereCount to know how many spheres to iterate over for intersection tests.
  const sceneParamsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    sceneParamsBuffer,
    0,
    new Uint32Array([sphereCount, 0, 0, 0])
  );
  // Load and compile compute shader module
  const computeResp = await fetch("./wgsl/pathtrace.wgsl");
  const computeCode = await computeResp.text();
  const computeModule = device.createShaderModule({ code: computeCode });
  computeModule.getCompilationInfo().then((info) => {
    if (info.messages.length) console.log("compute compile:", info.messages);
  });

  // Compute bind group layout MUST match the shader's @group(0) bindings.
  // Binding indices and types (order) are significant and must stay in sync
  // with `webgpu/wgsl/pathtrace.wgsl`:
  // 0 -> displayOut (storage texture rgba8unorm, write)
  // 1 -> CameraUBO (uniform buffer)
  // 2 -> seeds (storage buffer of u32 PRNG states)
  // 3 -> accumSrc (sampled texture, previous accumulation sum)
  // 4 -> accumDst (storage texture rgba32float, write)
  // 5 -> sampleCountUBO (uniform buffer: current sample count)
  // 6 -> spheres (read-only storage buffer)
  // 7 -> sceneParams (uniform buffer: sphere count)
  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: "write-only",
          format: "rgba8unorm",
          viewDimension: "2d",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", minBindingSize: cameraBufferSize },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: "write-only",
          format: accumFormat,
          viewDimension: "2d",
        },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", minBindingSize: 16 },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", minBindingSize: 16 },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [computeBindGroupLayout],
  });
  await device.pushErrorScope("validation");
  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: computeModule, entryPoint: "main" },
  });
  const scopeErr = await device.popErrorScope();
  if (scopeErr)
    console.error("Compute pipeline creation validation error:", scopeErr);

  // This sampler configures how the render pipeline samples the
  // display texture when drawing to the screen
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });
  const renderShaderCode = `
      @group(0) @binding(0) var myTexture : texture_2d<f32>;
      @group(0) @binding(1) var mySampler : sampler;
      struct VSOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
      @vertex fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
        var positions = array<vec2<f32>,3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
        var out: VSOut;
        out.pos = vec4<f32>(positions[vi],0.0,1.0);
        out.uv = positions[vi]*0.5 + vec2<f32>(0.5,0.5);
        return out;
      }
      @fragment fn fs_main(in_: VSOut) -> @location(0) vec4<f32> {
        let uv = clamp(in_.uv, vec2<f32>(0.0,0.0), vec2<f32>(1.0,1.0));
        let c = textureSample(myTexture, mySampler, uv);
        return vec4<f32>(c.rgb,1.0);
      }
    `;
  const renderModule = device.createShaderModule({ code: renderShaderCode });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: renderModule, entryPoint: "vs_main" },
    fragment: {
      module: renderModule,
      entryPoint: "fs_main",
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: "triangle-list" },
  });

  const renderBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: displayTex.createView() },
      { binding: 1, resource: sampler },
    ],
  });

  // Compute dispatch size: one workgroup per 8x8 pixel block
  // Compute how many compute workgroups to launch so the compute shader covers every pixel.
  // Workgroup = a group of shader invocations (threads) that execute together on one compute
  // unit and can share fast on-chip memory and synchronize.
  const workgroupSizeX = 8,
    workgroupSizeY = 8;
  const dispatchX = Math.ceil(width / workgroupSizeX);
  const dispatchY = Math.ceil(height / workgroupSizeY);

  // frame: run one progressive sample for the whole image.
  // After TARGET_SAMPLES are accumulated (and the GPU work for that frame completed),
  // print a single start/end/elapsed message and stop the loop.
  // This will call itself via requestAnimationFrame until done.
  async function frame() {
    // - Timing: the host records `renderStartTime` when the first compute work is
    //   submitted and prints a single start/finish/elapsed message when
    //   `TARGET_SAMPLES` have completed. The loop then stops.
    //     // set start time at the first compute submission (do not log now)
    if (renderStartTime === null) {
      renderStartTime = performance.now();
      renderStartISO = new Date().toISOString();
    }

    frameIndex += 1;

    device.queue.writeBuffer(
      sampleCountBuffer,
      0,
      new Uint32Array([sampleCount, 0, 0, 0])
    );

    const computeBindGroup = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        { binding: 0, resource: displayTex.createView() },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: seedBuffer } },
        { binding: 3, resource: accumSrc.createView() },
        { binding: 4, resource: accumDst.createView() },
        { binding: 5, resource: { buffer: sampleCountBuffer } },
        { binding: 6, resource: { buffer: sphereBuffer } },
        { binding: 7, resource: { buffer: sceneParamsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(computePipeline);
    cpass.setBindGroup(0, computeBindGroup);
    cpass.dispatchWorkgroups(dispatchX, dispatchY);
    cpass.end();

    const view = context.getCurrentTexture().createView();
    const rpass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    rpass.setPipeline(pipeline);
    rpass.setBindGroup(0, renderBindGroup);
    rpass.draw(3, 1, 0, 0);
    rpass.end();

    device.queue.submit([encoder.finish()]);

    // await GPU work completion where available; fallback to a short timeout otherwise.
    if (
      device.queue &&
      typeof device.queue.onSubmittedWorkDone === "function"
    ) {
      try {
        await device.queue.onSubmittedWorkDone();
      } catch (e) {
        await new Promise((r) => setTimeout(r, 16));
      }
    } else {
      await new Promise((r) => setTimeout(r, 16));
    }

    // advance accumulation state for progressive rendering
    sampleCount += SAMPLES_PER_FRAME; // SAMPLES_PER_FRAME == 1
    const tmp = accumSrc;
    accumSrc = accumDst;
    accumDst = tmp;

    // if we've reached or exceeded the target, log start/end/elapsed once and stop.
    if (!finished && sampleCount >= TARGET_SAMPLES) {
      const finishTime = performance.now();
      const finishISO = new Date().toISOString();
      const elapsedMs = finishTime - renderStartTime;
      console.log(
        `Render start: ${renderStartISO}\nRender finish: ${finishISO}\nElapsed: ${(
          elapsedMs / 1000
        ).toFixed(3)} s`
      );
      finished = true;
      return; // stop the frame loop
    }

    // otherwise continue rendering progressively
    requestAnimationFrame(frame);
  }

  // start the progressive render loop
  requestAnimationFrame(frame);
}

// Light the fuse!!!
init();
