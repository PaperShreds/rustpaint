mod app;
mod document;
mod export;
mod files;
mod mesh;
mod painting;
mod project;
mod recovery;
mod renderer;
mod settings;
mod ui;

use app::RustPaintApp;
use eframe::egui;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_title("RustPaint"),
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
