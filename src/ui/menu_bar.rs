use eframe::egui;

pub enum MenuAction {
    OpenModel,
    OpenProject,
    SaveProject,
    SaveProjectAs,
}

const SAVE: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
const SAVE_AS: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::S,
);

pub fn show(ui: &mut egui::Ui) -> Option<MenuAction> {
    // Check the more specific shortcut first: Ctrl+Shift+S also matches Ctrl+S.
    let mut action = ui.input_mut(|i| {
        if i.consume_shortcut(&SAVE_AS) {
            Some(MenuAction::SaveProjectAs)
        } else if i.consume_shortcut(&SAVE) {
            Some(MenuAction::SaveProject)
        } else {
            None
        }
    });

    egui::Panel::top("menu_bar").show(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open model...").clicked() {
                    action = Some(MenuAction::OpenModel);
                }
                if ui.button("Open project...").clicked() {
                    action = Some(MenuAction::OpenProject);
                }
                ui.separator();
                let shortcut = ui.ctx().format_shortcut(&SAVE);
                if ui
                    .add(egui::Button::new("Save project").shortcut_text(shortcut))
                    .clicked()
                {
                    action = Some(MenuAction::SaveProject);
                }
                let shortcut = ui.ctx().format_shortcut(&SAVE_AS);
                if ui
                    .add(egui::Button::new("Save project as...").shortcut_text(shortcut))
                    .clicked()
                {
                    action = Some(MenuAction::SaveProjectAs);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        });
    });

    action
}
