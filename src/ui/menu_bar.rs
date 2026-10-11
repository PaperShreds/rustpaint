use eframe::egui;

pub enum MenuAction {
    OpenModel,
    OpenProject,
    SaveProject,
    SaveProjectAs,
    ExportTextures,
    Undo,
    Redo,
}

const SAVE: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
const SAVE_AS: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::S,
);
const UNDO: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
const REDO: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::Z,
);
const REDO_ALT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);

/// Which menu entries are currently usable.
pub struct MenuState {
    pub can_undo: bool,
    pub can_redo: bool,
}

pub fn show(ui: &mut egui::Ui, state: &MenuState) -> Option<MenuAction> {
    // Check the more specific shortcuts first: Ctrl+Shift+S also matches Ctrl+S.
    let mut action = ui.input_mut(|i| {
        if i.consume_shortcut(&SAVE_AS) {
            Some(MenuAction::SaveProjectAs)
        } else if i.consume_shortcut(&SAVE) {
            Some(MenuAction::SaveProject)
        } else if i.consume_shortcut(&REDO) || i.consume_shortcut(&REDO_ALT) {
            Some(MenuAction::Redo)
        } else if i.consume_shortcut(&UNDO) {
            Some(MenuAction::Undo)
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
                if ui.button("Export textures...").clicked() {
                    action = Some(MenuAction::ExportTextures);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                let shortcut = ui.ctx().format_shortcut(&UNDO);
                let undo = egui::Button::new("Undo").shortcut_text(shortcut);
                if ui.add_enabled(state.can_undo, undo).clicked() {
                    action = Some(MenuAction::Undo);
                }
                let shortcut = ui.ctx().format_shortcut(&REDO);
                let redo = egui::Button::new("Redo").shortcut_text(shortcut);
                if ui.add_enabled(state.can_redo, redo).clicked() {
                    action = Some(MenuAction::Redo);
                }
            });
        });
    });

    action
}
