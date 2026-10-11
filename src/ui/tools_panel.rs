use crate::painting::brush::{self, Brush, FillMode, Tool};
use eframe::egui;

pub fn show(ui: &mut egui::Ui, brush: &mut Brush) {
    egui::Panel::left("tools_panel")
        .resizable(true)
        .default_size(240.0)
        .show(ui, |ui| {
            ui.heading("Brush");
            ui.separator();

            ui.horizontal(|ui| {
                ui.selectable_value(&mut brush.tool, Tool::Paint, "Paint");
                ui.selectable_value(&mut brush.tool, Tool::Erase, "Erase")
                    .on_hover_text("Remove paint from the selected layer");
                ui.selectable_value(&mut brush.tool, Tool::Fill, "Fill")
                    .on_hover_text("Click the model to fill the selected layer");
            });

            if brush.tool == Tool::Fill {
                egui::ComboBox::from_label("Fill")
                    .selected_text(brush.fill_mode.name())
                    .show_ui(ui, |ui| {
                        for mode in FillMode::ALL {
                            ui.selectable_value(&mut brush.fill_mode, mode, mode.name());
                        }
                    })
                    .response
                    .on_hover_text(
                        "Whole layer: every material the layer paints on.\n\
                         Clicked material: the material under the cursor.\n\
                         Clicked UV island: the connected UV piece under the cursor.",
                    );
            } else {
                // Zooming may take the size past the slider's end; keep it.
                ui.add(
                    egui::Slider::new(&mut brush.size, brush::MIN_SIZE..=brush::SLIDER_MAX_SIZE)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Size"),
                );
                ui.add(egui::Slider::new(&mut brush.hardness, 0.0..=1.0).text("Hardness"));
            }
            ui.add(egui::Slider::new(&mut brush.opacity, 0.0..=1.0).text("Opacity"));

            ui.add_space(8.0);
            ui.label("Channels")
                .on_hover_text("Tick the channels a stroke paints, and the value it puts there");
            egui::Grid::new("brush_channels")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.checkbox(&mut brush.paint_color, "Base color");
                    ui.color_edit_button_rgb(&mut brush.color);
                    ui.end_row();
                    ui.checkbox(&mut brush.paint_roughness, "Roughness");
                    ui.add(egui::Slider::new(&mut brush.roughness, 0.0..=1.0));
                    ui.end_row();
                    ui.checkbox(&mut brush.paint_metallic, "Metallic");
                    ui.add(egui::Slider::new(&mut brush.metallic, 0.0..=1.0));
                    ui.end_row();
                });
            ui.add_space(8.0);
            ui.checkbox(&mut brush.screen_space, "Screen-space brush")
                .on_hover_text(
                    "Project the brush flat from the screen instead of following the surface",
                );
            ui.checkbox(&mut brush.paint_over_edges, "Paint over edges")
                .on_hover_text(
                    "When off, a stroke stays on the face where it started. \
                     Faces are bounded by the model's hard (flat-shaded) edges.",
                );
            ui.checkbox(&mut brush.keep_size_on_model, "Keep size on model")
                .on_hover_text(
                    "The brush covers the same area of the model when you zoom, and on \
                     near and far parts alike. When off, it keeps its size on screen, \
                     so it covers more of the model further away.",
                );
        });
}
