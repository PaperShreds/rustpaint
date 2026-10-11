# RustPaint

RustPaint is an early-stage 3D texturing application written in Rust. Its goal is
to grow into a straightforward, crash-resistant alternative for painting textures
on 3D models.

## Current status

The app provides a 3D viewport, loads `.glb` and `.gltf` models, and lets you
paint the base color of the selected texture set. Projects can be saved to and
reopened from `.rustpaint` files.

Viewport controls:

- **Left drag:** paint
- **Right drag** or **Alt + left drag:** orbit
- **Middle drag** or **Shift + right drag:** pan
- **Scroll:** zoom

By default the brush is a sphere on the model's surface: it keeps the same size
on the model and wraps around edges. **Screen-space brush** instead projects a
flat circle from the screen onto whatever is visible, so it stretches on surfaces
seen at an angle.

Turn off **Paint over edges** to keep a stroke on the face where it started.
A face ends at the model's hard edges (where its shading normals split), so a
cube has six faces while a smooth-shaded sphere is one.

Paint goes into the layer selected in the right-hand panel. Each layer has a
visibility checkbox, a thumbnail, a name (double-click to rename), and a
dropdown of the material slots it paints on; one stroke can cross several
slots. The bottom **Base** layer holds the model's own textures: it can be
hidden but not painted on, so the original stays intact. Use **Erase** in the
brush panel to remove paint from a layer. Materials without a texture get a
blank 2048×2048 canvas in their base color.

Undo and redo strokes and layer changes with **Ctrl+Z** and **Ctrl+Shift+Z**
(or **Ctrl+Y**).
History isn't saved with the project.

**File > Export textures...** writes each material slot's base color, with its
visible layers combined, as a PNG named `<model>_<material>_BaseColor.png`. It
asks before replacing existing files.

Blend modes, masks, other material channels (roughness, metallic, normal), and
autosave are not implemented yet, and the project file format may still change.
Avoid using it as the only copy of important work.

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
