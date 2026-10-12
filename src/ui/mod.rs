pub mod history_panel;
pub mod layers_panel;
pub mod menu_bar;
pub mod status_bar;
pub mod tools_panel;
pub mod viewport;

/// Makes sliders and drop-downs grow with the panel they are in, so a wider
/// panel doesn't leave them bunched at the left.
pub fn fit_widths(ui: &mut eframe::egui::Ui) {
    let width = (ui.available_width() * 0.45).clamp(80.0, 400.0);
    ui.spacing_mut().slider_width = width;
    ui.spacing_mut().combo_width = width;
}
