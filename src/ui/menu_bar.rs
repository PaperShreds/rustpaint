use crate::renderer::environment::{BuiltIn, EnvironmentSource};
use crate::renderer::viewport::{EnvironmentSettings, ViewMode};
use eframe::egui;
use std::path::PathBuf;

pub enum MenuAction {
    OpenModel,
    OpenProject,
    OpenRecent(PathBuf),
    ClearRecent,
    SaveProject,
    SaveProjectAs,
    ExportTextures,
    ExportModel,
    Undo,
    Redo,
    SetView(ViewMode),
    LoadEnvironment,
    SetEnvironment(BuiltIn),
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
pub struct MenuState<'a> {
    pub can_undo: bool,
    pub can_redo: bool,
    pub view_mode: ViewMode,
    pub recent_files: &'a [PathBuf],
    pub environment_source: &'a EnvironmentSource,
}

pub fn show(
    ui: &mut egui::Ui,
    state: &MenuState<'_>,
    environment: &mut EnvironmentSettings,
) -> Option<MenuAction> {
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
                ui.add_enabled_ui(!state.recent_files.is_empty(), |ui| {
                    ui.menu_button("Open recent", |ui| {
                        for path in state.recent_files {
                            let name = path.file_name().map_or_else(
                                || path.display().to_string(),
                                |name| name.to_string_lossy().into_owned(),
                            );
                            let button = ui.button(name).on_hover_text(path.display().to_string());
                            if button.clicked() {
                                action = Some(MenuAction::OpenRecent(path.clone()));
                            }
                        }
                        ui.separator();
                        if ui.button("Clear list").clicked() {
                            action = Some(MenuAction::ClearRecent);
                        }
                    });
                });
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
                if ui
                    .button("Export model (.glb)...")
                    .on_hover_text("The model with its painted textures, in one file")
                    .clicked()
                {
                    action = Some(MenuAction::ExportModel);
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
            ui.menu_button("View", |ui| {
                ui.label("Show in viewport");
                for mode in ViewMode::ALL {
                    if ui.radio(state.view_mode == mode, mode.name()).clicked() {
                        action = Some(MenuAction::SetView(mode));
                    }
                }
                ui.separator();
                ui.label("Environment");
                for built_in in BuiltIn::ALL {
                    let selected =
                        *state.environment_source == EnvironmentSource::BuiltIn(built_in);
                    if ui.radio(selected, built_in.name()).clicked() {
                        action = Some(MenuAction::SetEnvironment(built_in));
                    }
                }
                if let EnvironmentSource::File(path) = state.environment_source {
                    let _ = ui
                        .radio(true, state.environment_source.name())
                        .on_hover_text(path.display().to_string());
                }
                if ui.button("Load HDRI...").clicked() {
                    action = Some(MenuAction::LoadEnvironment);
                }
                ui.add(
                    egui::Slider::new(&mut environment.rotation_degrees, 0.0..=360.0)
                        .text("Rotation")
                        .suffix("°"),
                );
                ui.add(egui::Slider::new(&mut environment.intensity, 0.0..=4.0).text("Intensity"));
                ui.checkbox(&mut environment.show_background, "Show as background");
                ui.add_enabled(
                    environment.show_background,
                    egui::Slider::new(&mut environment.background_blur, 0.0..=1.0)
                        .text("Background blur"),
                );
            });
        });
    });

    action
}
