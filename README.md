# ray-trace-rest-of-your-life

This repo is a WebGPU / WebGPU Shader Language(WGSL) implemenetation of Ray Tracing in One Weekend by Peter Shirley, Trevor David Black, Steve Hollasch.

https://raytracing.github.io/books/RayTracingInOneWeekend.html

With the assistance of Github Copilot, and using the GPT-5 mini model, I ported this into WebGPU and WGSL from a Python implementation that I had previously written. I have learned a lot about Ray Tracers and a lot about WebGPU/WGSL along the way.

To see the code in action:

Because WebGPU requires secure context / localhost, distributing a single index.html + \*.wgsl + main.js is fine, but the user must run a local server (no bundler needed). This still qualifies as a static site—just not double-click open.

Python 3

# from project root

```
python3 -m http.server 8000
```

# then open http://localhost:8000 in Chrome

Node (http-server)

```
npm install -g http-server
http-server -c-1 . -p 8000
```

# open http://localhost:8000 in Chrome

# web-gpu-ray-tracer
