mod app;
mod document;
mod export;
mod files;
mod generator;
mod mesh;
mod painting;
mod project;
mod recovery;
mod renderer;
mod settings;
mod ui;

use app::RustPaintApp;
use eframe::egui;

/// The window icon, drawn from `assets/logo/logo.svg` (see its README).
fn window_icon() -> Option<egui::IconData> {
    let png = include_bytes!("../assets/logo/icon.png");
    let image = image::load_from_memory(png).ok()?.into_rgba8();
    Some(egui::IconData {
        width: image.width(),
        height: image.height(),
        rgba: image.into_raw(),
    })
}

fn main() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 720.0])
        .with_title("RustPaint")
        .with_app_id("rustpaint");
    if let Some(icon) = window_icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: renderer::viewport::DEPTH_BUFFER_BITS, // new
        ..Default::default()
    };

    eframe::run_native(
        "RustPaint",
        options,
        Box::new(|cc| Ok(Box::new(RustPaintApp::new(cc)))),
    )
}
