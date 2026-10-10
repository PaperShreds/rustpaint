# RustPaint

RustPaint is an early-stage 3D texturing application written in Rust. Its goal is
to grow into a straightforward, crash-resistant alternative for painting textures
on 3D models.

## Current status

The app currently provides a 3D viewport and can load `.glb` and `.gltf` models,
showing their meshes and base-color materials. You can orbit the camera by
dragging in the viewport and zoom with the scroll wheel.

The brush controls, material selector, and layer list are early UI scaffolding:
painting and layer editing are not implemented yet. Saving, project files, and
autosave are also not available yet. Avoid using it as the only copy of important
work.

## Run

Install the stable Rust toolchain, then run:

```sh
git clone https://github.com/PaperShreds/rustpaint.git
cd rustpaint
cargo run
```

Use **File > Open model...** to open a `.glb` or `.gltf` file. On supported
platforms, you can also drag a model file into the application window.

## Development

Build and run the test suite with:

```sh
cargo build
cargo test
```

## Next milestone

The next useful milestone is a small, complete painting workflow: paint on a
model's texture, then save and reopen that work. Establishing a project format
and reliable save behavior early will also provide a foundation for autosave
and crash recovery.
