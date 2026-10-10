use eframe::egui;

pub fn show(
    ui: &mut egui::Ui,
    materials: &[String],
    selected_material: &mut usize,
    layers: &mut Vec<String>,
    selected_layer: &mut usize,
) {
    egui::Panel::right("layers_panel")
        .resizable(true)
        .default_size(240.0)
        .show(ui, |ui| {
            ui.heading("Texture sets");
            ui.separator();
            for (i, name) in materials.iter().enumerate() {
                if ui.selectable_label(*selected_material == i, name).clicked() {
                    *selected_material = i;
                }
            }

            ui.add_space(16.0);
            ui.heading("Layers");
            ui.separator();
            for (i, name) in layers.iter().enumerate() {
                if ui.selectable_label(*selected_layer == i, name).clicked() {
                    *selected_layer = i;
                }
            }
            ui.separator();
            if ui.button("+ Add layer").clicked() {
                layers.push(format!("Layer {}", layers.len() + 1));
                *selected_layer = layers.len() - 1;
            }
        });
}