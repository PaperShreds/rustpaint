use eframe::egui;

pub fn show(ui: &mut egui::Ui, text: &str) {
    egui::Panel::bottom("status_bar").show(ui, |ui| {
        ui.label(text);
    });
}
