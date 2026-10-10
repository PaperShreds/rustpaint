use crate::renderer::camera::OrbitCamera;
use crate::renderer::viewport::ViewportCallback;
use eframe::egui;
use eframe::egui_wgpu;

pub fn show(ui: &mut egui::Ui, camera: &mut OrbitCamera) {
    egui::CentralPanel::default().show(ui, |ui| {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

        if response.dragged() {
            let delta = response.drag_motion();
            camera.orbit(delta.x, delta.y);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            camera.zoom(scroll);
        }

        let aspect = rect.width() / rect.height().max(1.0);
        let view_proj = camera.view_proj(aspect).to_cols_array_2d();

        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(30));
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            ViewportCallback { view_proj },
        ));
    });
}
