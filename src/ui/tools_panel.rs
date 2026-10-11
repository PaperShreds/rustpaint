use crate::painting::brush::{self, Brush, FillMode, Tool};
use eframe::egui;

const BUTTON_HEIGHT: f32 = 28.0;
const ICON_SIZE: f32 = 18.0;
const ERASE_ICON: egui::ImageSource<'static> = egui::include_image!("../../assets/icons/erase.svg");

pub fn show(ui: &mut egui::Ui, brush: &mut Brush) {
    egui::Panel::left("tools_panel")
        .resizable(true)
        .default_size(240.0)
        .show(ui, |ui| {
            ui.heading("Brush");
            ui.separator();

            // Tools in two columns of equal width.
            let spacing = ui.spacing().item_spacing.x;
            let half = egui::vec2((ui.available_width() - spacing) / 2.0, BUTTON_HEIGHT);
            egui::Grid::new("tools").num_columns(2).show(ui, |ui| {
                for (index, tool) in Tool::ALL.into_iter().enumerate() {
                    let selected = brush.tool == tool;
                    let button = icon_button(ui, tool_icon(tool), tool.name(), selected);
                    if ui
                        .add(button.min_size(half))
                        .on_hover_text(tool_hint(tool))
                        .clicked()
                    {
                        brush.tool = tool;
                    }
                    if index % 2 == 1 {
                        ui.end_row();
                    }
                }
                // Erase is a toggle on top of the tool, in its own row.
                ui.add_enabled_ui(brush.tool.can_erase(), |ui| {
                    let erase = icon_button(ui, ERASE_ICON, "Erase", brush.erases());
                    if ui
                        .add(erase.min_size(half))
                        .on_hover_text("Paint and Fill remove paint from the layer instead")
                        .on_disabled_hover_text("Smear and Blur only move paint around")
                        .clicked()
                    {
                        brush.erase = !brush.erase;
                    }
                });
                ui.end_row();
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
            let moves_paint = matches!(brush.tool, Tool::Smear | Tool::Blur);
            let strength = if moves_paint { "Strength" } else { "Opacity" };
            ui.add(egui::Slider::new(&mut brush.opacity, 0.0..=1.0).text(strength));

            ui.add_space(8.0);
            ui.label("Channels").on_hover_text(
                "Tick the channels a stroke works on, and the value Paint and Fill put there",
            );
            // Smear and blur rework what is there; the values don't apply.
            let values = !moves_paint && !brush.erases();
            egui::Grid::new("brush_channels")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.checkbox(&mut brush.paint_color, "Base color");
                    ui.add_enabled_ui(values, |ui| ui.color_edit_button_rgb(&mut brush.color));
                    ui.end_row();
                    ui.checkbox(&mut brush.paint_roughness, "Roughness");
                    ui.add_enabled(values, egui::Slider::new(&mut brush.roughness, 0.0..=1.0));
                    ui.end_row();
                    ui.checkbox(&mut brush.paint_metallic, "Metallic");
                    ui.add_enabled(values, egui::Slider::new(&mut brush.metallic, 0.0..=1.0));
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

fn tool_hint(tool: Tool) -> &'static str {
    match tool {
        Tool::Paint => "Paint the ticked channels onto the selected layer",
        Tool::Smear => "Push the layer's paint along the stroke",
        Tool::Blur => "Soften the layer's paint under the brush",
        Tool::Fill => "Click the model to fill the selected layer",
    }
}

fn tool_icon(tool: Tool) -> egui::ImageSource<'static> {
    match tool {
        Tool::Paint => egui::include_image!("../../assets/icons/brush.svg"),
        Tool::Smear => egui::include_image!("../../assets/icons/smear.svg"),
        Tool::Blur => egui::include_image!("../../assets/icons/blur.svg"),
        Tool::Fill => egui::include_image!("../../assets/icons/fill.svg"),
    }
}

/// A toggle-style button with an icon and a label. The icons are drawn in
/// white, so tinting them gives them the same colour as the text.
fn icon_button<'a>(
    ui: &egui::Ui,
    icon: egui::ImageSource<'a>,
    label: &'a str,
    selected: bool,
) -> egui::Button<'a> {
    let visuals = ui.visuals();
    let tint = if !ui.is_enabled() {
        visuals.weak_text_color()
    } else if selected {
        visuals.selection.stroke.color
    } else {
        visuals.text_color()
    };
    let icon = egui::Image::new(icon)
        .fit_to_exact_size(egui::vec2(ICON_SIZE, ICON_SIZE))
        .tint(tint);
    egui::Button::selectable(selected, (icon, label))
}
