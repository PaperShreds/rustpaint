use crate::painting::brush::Brush;
use eframe::egui;

pub fn show(ui: &mut egui::Ui, brush: &mut Brush) {
    egui::Panel::left("tools_panel")
        .resizable(true)
        .default_size(240.0)
        .show(ui, |ui| {
            ui.heading("Brush");
            ui.separator();

            ui.add(egui::Slider::new(&mut brush.size, 1.0..=512.0).text("Size"));
            ui.add(egui::Slider::new(&mut brush.hardness, 0.0..=1.0).text("Hardness"));
            ui.add(egui::Slider::new(&mut brush.opacity, 0.0..=1.0).text("Opacity"));

            ui.horizontal(|ui| {
                ui.label("Color");
                ui.color_edit_button_rgb(&mut brush.color);
            });
        });
}
