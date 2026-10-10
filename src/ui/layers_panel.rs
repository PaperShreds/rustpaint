use eframe::egui;

pub fn show(ui: &mut egui::Ui, layers: &mut Vec<String>, selected: &mut usize) {
    egui::Panel::right("layers_panel")
        .resizable(true)
        .default_size(240.0)
        .show(ui, |ui| {
            ui.heading("Layers");
            ui.separator();

            for (i, name) in layers.iter().enumerate() {
                if ui.selectable_label(*selected == i, name).clicked() {
                    *selected = i;
                }
            }

            ui.separator();
            if ui.button("+ Add layer").clicked() {
                layers.push(format!("Layer {}", layers.len() + 1));
                *selected = layers.len() - 1;
            }
        });
}
