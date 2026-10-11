use crate::painting::history::History;
use eframe::egui;

/// Lists the undo history at the bottom of `ui`. Clicking a step undoes or
/// redoes up to it. Returns the position picked: how many steps to keep applied.
pub fn show(ui: &mut egui::Ui, history: &History) -> Option<usize> {
    let mut picked = None;
    egui::Panel::bottom("history_panel")
        .resizable(true)
        .default_size(200.0)
        .show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("History");
                let (used, budget) = history.memory();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("{} / {} MB", used.div_ceil(1 << 20), budget >> 20))
                        .on_hover_text(
                            "Memory the undo steps use. The oldest steps are dropped \
                             when it runs out.",
                        );
                });
            });
            ui.separator();

            let position = history.position();
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    let start = if history.dropped() > 0 {
                        "Oldest kept step"
                    } else {
                        "Start"
                    };
                    let rows = std::iter::once(start).chain(history.labels());
                    for (index, label) in rows.enumerate() {
                        // Row `index` is the state with `index` steps applied.
                        let text = if index > position {
                            // Undone: redo brings it back, a new edit discards it.
                            egui::RichText::new(label).weak()
                        } else {
                            egui::RichText::new(label)
                        };
                        let row = ui.add(
                            egui::Button::selectable(index == position, text)
                                .frame_when_inactive(false),
                        );
                        if row.clicked() && index != position {
                            picked = Some(index);
                        }
                    }
                });
        });
    picked
}
