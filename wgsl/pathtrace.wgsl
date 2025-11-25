// WGSL compute shader: iterative path tracer (no recursion)
//
// High-level mapping to "Ray Tracing in One Weekend":
// - Primary ray generation: jittered subpixel sampling (u,v) per sample.
// - Monte Carlo integration: one stochastic sample per dispatch (host
//   controls `SAMPLES_PER_FRAME`) and accumulation into a high-precision
//   `rgba32float` texture to avoid precision loss when summing many samples.
// - Materials: Lambertian (diffuse / cosine sampling), Metal (reflect +
//   fuzz), Dielectric (Schlick + refract), and Emissive (return emission
//   and terminate). Implemented iteratively inside a `for` loop up to
//   `MAX_DEPTH` bounces.
// - RNG: per-pixel xorshift32 stored in a storage buffer. The CPU seeds the
//   buffer initially; the shader advances and stores back the state each
//   invocation so subsequent frames continue the stream.
// - Depth-of-field (DoF): the Camera UBO's padding field (`_pad3`) carries a
//   `lensRadius`. When > 0 the shader samples a disk to offset ray origins,
//   implementing thin-lens defocus per the book (lens -> focus distance).
// - Accumulation & Display: shader reads previous sum from `accumSrc`, adds
//   this sample, writes new sum to `accumDst`. It also writes an averaged and
//   gamma-corrected value to `displayOut` so the UI shows progressive
//   convergence.

// (no extensions enabled)

struct CameraUBO {
    origin : vec3<f32>,
    _pad0 : f32,
    lower_left : vec3<f32>,
    _pad1 : f32,
    horizontal : vec3<f32>,
    _pad2 : f32,
    vertical : vec3<f32>,
    _pad3 : f32, // holds lens radius when written from JS
    width: u32,
    height: u32,
    _pad4: u32,
    _pad5: u32,
};

@group(0) @binding(1) var<uniform> cam : CameraUBO;

// per-pixel 32-bit xorshift PRNG state (one u32 per pixel)
@group(0) @binding(2) var<storage, read_write> seeds : array<u32>;

// previous accumulation (linear floating sum) as sampled texture
@group(0) @binding(3) var accumSrc : texture_2d<f32>;

// output accumulation (rgba32float write-only)
@group(0) @binding(4) var accumDst : texture_storage_2d<rgba32float, write>;

// output display (rgba8unorm write-only) — shader will write final gamma-encoded color here
@group(0) @binding(0) var displayOut : texture_storage_2d<rgba8unorm, write>;

// sample count UBO (single u32) — number of samples already accumulated
struct SampleCountUBO { samples : u32, _pad0 : u32, _pad1 : u32, _pad2 : u32 };
@group(0) @binding(5) var<uniform> sampleCountUBO : SampleCountUBO;

// spheres stored as contiguous floats; main.js packs 16 floats per sphere
@group(0) @binding(6) var<storage, read> spheres : array<f32>;

// scene params: sphere count in the first uint
struct SceneParams { sphereCount : u32, _pad0 : u32, _pad1 : u32, _pad2 : u32 };
@group(0) @binding(7) var<uniform> sceneParams : SceneParams;

const PI = 3.14159265359;
const MAX_DEPTH = 50;

// RNG: xorshift32 per-pixel; returns float in [0,1)
fn rng_next(idx : u32) -> f32 {
    var s = seeds[idx];
    // xorshift32
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    seeds[idx] = s;
    // convert to [0,1) using full 32-bit range
    return f32(s) / 4294967296.0;
}

// utility vec ops (WGSL lacks some convenience)
// for metal
fn reflect(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}
// for dielectric
fn refract(uv: vec3<f32>, n: vec3<f32>, etai_over_etat: f32) -> vec3<f32> {
    let cos_theta = min(dot(-uv, n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}
// Schlick approximation for reflectance
fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

// random point in unit sphere (used for diffuse)
fn random_in_unit_sphere(ix: u32) -> vec3<f32> {
    var p = vec3<f32>(0.0, 0.0, 0.0);
    loop {
        let x = 2.0 * rng_next(ix) - 1.0;
        let y = 2.0 * rng_next(ix) - 1.0;
        let z = 2.0 * rng_next(ix) - 1.0;
        p = vec3<f32>(x, y, z);
        if (dot(p,p) < 1.0) { break; }
    }
    return p;
}
fn unit_vector(v: vec3<f32>) -> vec3<f32> {
    return normalize(v);
}

// sample a point uniformly on a unit disk using two RNG calls (returns vec2)
fn random_in_unit_disk(ix: u32) -> vec2<f32> {
    let r1 = rng_next(ix);
    let r2 = rng_next(ix);
    let theta = 2.0 * PI * r1;
    let r = sqrt(r2);
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// read sphere fields (packed as 16 floats per sphere)
// layout (per main.js): [cx,cy,cz,r], [albedo.r,albedo.g,albedo.b,kind], [fuzz,ref_idx,emit.r,emit.g], [emit.b, pad..., ...]
fn sphere_get_center(i: u32) -> vec3<f32> {
    let base = i * 16u;
    return vec3<f32>(spheres[base + 0u], spheres[base + 1u], spheres[base + 2u]);
}
fn sphere_get_radius(i: u32) -> f32 {
    return spheres[i * 16u + 3u];
}
fn sphere_get_albedo(i: u32) -> vec3<f32> {
    return vec3<f32>(spheres[i * 16u + 4u], spheres[i * 16u + 5u], spheres[i * 16u + 6u]);
}
fn sphere_get_kind(i: u32) -> u32 {
    return u32(spheres[i * 16u + 7u]);
}
fn sphere_get_fuzz(i: u32) -> f32 {
    return spheres[i * 16u + 8u];
}
fn sphere_get_refidx(i: u32) -> f32 {
    return spheres[i * 16u + 9u];
}
fn sphere_get_emission(i: u32) -> vec3<f32> {
    return vec3<f32>(spheres[i * 16u + 10u], spheres[i * 16u + 11u], spheres[i * 16u + 12u]);
}

// sphere hit test: returns t, normal, front_face flag
fn hit_sphere(i: u32, ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32)
    -> vec4<f32> // (t, nx, ny, nz) or ( -1, 0,0,0 ) if miss
{
    let center = sphere_get_center(i);
    let radius = sphere_get_radius(i);
    let oc = ro - center;
    let a = dot(rd, rd);
    let half_b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        }
    }
    let p = ro + root * rd;
    let outward_normal = (p - center) / radius;
    // store normal as returned vector (we compute front_face in caller if needed)
    return vec4<f32>(root, outward_normal.x, outward_normal.y, outward_normal.z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ix = gid.x;
    let iy = gid.y;
    if (ix >= cam.width || iy >= cam.height) { return; }

    // pixel index into seeds buffer
    let pixIdx = ix + iy * cam.width;

    // generate jittered subpixel coordinates (Monte Carlo sampling)
    let rx = rng_next(pixIdx);
    let ry = rng_next(pixIdx);

    // normalized u,v in [0,1]
    let u = (f32(ix) + rx) / max(1.0, f32(cam.width - 1u));
    let v = (f32(iy) + ry) / max(1.0, f32(cam.height - 1u));

    // compute focal point on the focal plane (camera UBO already encodes focus_dist)
    let focal_point = cam.lower_left + u * cam.horizontal + v * cam.vertical;
    let origin = cam.origin;
    var ro = origin;
    var rd = normalize(focal_point - origin);

    // Depth-of-field: sample lens (lens radius stored in cam._pad3)
    let lens_radius = cam._pad3;
    if (lens_radius > 0.0) {
        let rdisk = random_in_unit_disk(pixIdx) * lens_radius;
        // derive camera plane axes from horizontal/vertical stored in UBO
        let uvec = normalize(cam.horizontal);
        let vvec = normalize(cam.vertical);
        let offset = uvec * rdisk.x + vvec * rdisk.y;
        ro = origin + offset;
        rd = normalize(focal_point - ro);
    }

    // iterative path tracing loop (no recursion in WGSL) per RTIAW chapter
    var throughput = vec3<f32>(1.0, 1.0, 1.0); // accumulated albedo multiplier
    var radiance = vec3<f32>(0.0, 0.0, 0.0);

    var t_min = 0.001;
    var t_max = 1e9;

    // bounce loop (the iterative replacement for recursive ray bounces, like the book implementation)
    var depth: i32 = 0;
    loop {
        if (depth >= MAX_DEPTH) {
            break;
        }
        depth = depth + 1;

        // find closest hit among spheres
        var closest_t = 1e9;
        var hit_any = false;
        var hit_normal = vec3<f32>(0.0, 0.0, 0.0);
        var hit_point = vec3<f32>(0.0, 0.0, 0.0);
        var hit_index: u32 = 0u;

        let scnt = sceneParams.sphereCount;
        var i: u32 = 0u;
        while (i < scnt) {
            let res = hit_sphere(i, ro, rd, t_min, closest_t);
            if (res.x > 0.0) {
                hit_any = true;
                closest_t = res.x;
                hit_normal = vec3<f32>(res.y, res.z, res.w);
                hit_index = i;
            }
            i = i + 1u;
        }

        if (!hit_any) {
            // background (simple gradient) — matches RTIAW background
            let unit_dir = unit_vector(rd);
            let t_bg = 0.5 * (unit_dir.y + 1.0);
            let c = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.5, 0.7, 1.0), t_bg);
            radiance = radiance + throughput * c;
            break;
        }

        // compute hit point and material
        hit_point = ro + closest_t * rd;
        // determine front face and adjust normal so it always points against ray
        var outward_normal = hit_normal;
        var front_face = dot(rd, outward_normal) < 0.0;
        var normal = select(-outward_normal, outward_normal, front_face);

        // fetch material properties
        let kind = sphere_get_kind(hit_index);
        let albedo = sphere_get_albedo(hit_index);
        let fuzz = sphere_get_fuzz(hit_index);
        let ref_idx = sphere_get_refidx(hit_index);
        let emission = sphere_get_emission(hit_index);

        // add emission (emissive sphere behaves as light)
        if (any(emission != vec3<f32>(0.0,0.0,0.0))) {
            radiance = radiance + throughput * emission;
            break;
        }

        // handle scattering by material kind (0: lambertian, 2: metal, 3: dielectric)
        if (kind == 0u) {
            // Lambertian: scatter out in random hemisphere (RTIAW)
            var scatter_dir = normal + unit_vector(random_in_unit_sphere(pixIdx));
            // catch degenerate scatter
            if (length(scatter_dir) < 1e-6) {
                scatter_dir = normal;
            }
            rd = normalize(scatter_dir);
            ro = hit_point;
            throughput = throughput * albedo;
            continue;
        } else if (kind == 2u) {
            // metal (reflective) with optional fuzz
            let reflected = reflect(normalize(rd), normal);
            rd = normalize(reflected + fuzz * random_in_unit_sphere(pixIdx));
            ro = hit_point;
            // if ray goes below surface, absorb
            if (dot(rd, normal) <= 0.0) {
                // absorbed — stop path
                break;
            }
            throughput = throughput * albedo;
            continue;
        } else if (kind == 3u) {
            // dielectric (glass) — fixed etai/etat selection and schlick usage
            let attenuation = vec3<f32>(1.0, 1.0, 1.0);
            // If front_face is true (ray is entering), etai_over_etat = 1.0 / ref_idx,
            // otherwise (inside->outside) etai_over_etat = ref_idx.
            let etai_over_etat = select(ref_idx, 1.0 / ref_idx, front_face);
            let unit_rd = unit_vector(rd);
            let cos_theta = min(dot(-unit_rd, normal), 1.0);
            let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            let cannot_refract = etai_over_etat * sin_theta > 1.0;

            // Use the material refractive index (ref_idx) for Schlick's approximation
            let reflect_prob = schlick(cos_theta, ref_idx);

            var direction = vec3<f32>(0.0, 0.0, 0.0);
            if (cannot_refract || reflect_prob > rng_next(pixIdx)) {
                direction = reflect(unit_rd, normal);
            } else {
                direction = refract(unit_rd, normal, etai_over_etat);
            }
            rd = normalize(direction);
            ro = hit_point;
            throughput = throughput * attenuation;
            continue;
        } else {
            // unknown kind: treat as diffuse
            var scatter_dir = normal + unit_vector(random_in_unit_sphere(pixIdx));
            if (length(scatter_dir) < 1e-6) {
                scatter_dir = normal;
            }
            rd = normalize(scatter_dir);
            ro = hit_point;
            throughput = throughput * albedo;
            continue;
        }
    } // end loop

    // final color for this sample (linear)
    let col = radiance;

    // read previous accumulation sum (linear) and add this sample
    let coord = vec2<i32>(i32(ix), i32(iy));
    var prev = textureLoad(accumSrc, coord, 0).xyz;
    var sum = prev + col;

    // write new accumulation sum into accumDst
    textureStore(accumDst, coord, vec4<f32>(sum.x, sum.y, sum.z, 0.0));

    // compute display color = average (sum / (n+1)), gamma-correct with simple sqrt
    let n = f32(sampleCountUBO.samples) + 1.0;
    var avg = sum / n;
    // gamma correction (approximate gamma 2)
    avg = vec3<f32>(sqrt(avg.x), sqrt(avg.y), sqrt(avg.z));

    // write to display texture (rgba8unorm auto-converted)
    textureStore(displayOut, coord, vec4<f32>(avg.x, avg.y, avg.z, 1.0));
}