use eframe::egui;

pub enum MenuAction {
    OpenModel,
}

pub fn show(ui: &mut egui::Ui) -> Option<MenuAction> {
    let mut action = None;

    egui::Panel::top("menu_bar").show(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open model...").clicked() {
                    action = Some(MenuAction::OpenModel);
                }
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        });
    });

    action
}
