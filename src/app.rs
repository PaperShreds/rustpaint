use crate::document::{Change, Dirty, Document, LayerProps};
use crate::export;
use crate::mesh::{self, Image, Mesh};
use crate::painting::brush::Brush;
use crate::painting::history::History;
use crate::painting::stroke::{Stroke, StrokePainter};
use crate::project;
use crate::renderer;
use crate::renderer::camera::OrbitCamera;
use crate::ui;
use crate::ui::layers_panel::{LayerAction, LayersPanelState};
use crate::ui::menu_bar::{MenuAction, MenuState};
use crate::ui::viewport::BrushCursor;
use eframe::egui;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const PROJECT_EXTENSION: &str = "rustpaint";
/// Untextured materials get a blank canvas of this size so they can be painted.
const DEFAULT_CANVAS_SIZE: u32 = 2048;

pub struct RustPaintApp {
    pub brush: Brush,
    pub camera: OrbitCamera,
    pub status: String,
    /// The model and its layers; painting edits this, then uploads the composites.
    document: Document,
    /// Id of the layer strokes go into.
    selected_layer: u64,
    layers_panel: LayersPanelState,
    thumbnails: HashMap<u64, egui::TextureHandle>,
    /// Bumped on every document change, so thumbnails know when to refresh.
    revision: u64,
    thumbnails_revision: Option<u64>,
    /// Settings of a layer before an opacity drag started, for its undo step.
    opacity_drag: Option<(u64, LayerProps)>,
    /// Screen-space acceleration structure for painting; rebuilt when the view changes.
    painter: Option<StrokePainter>,
    /// The stroke in progress while the mouse button is held.
    stroke: Option<Stroke>,
    history: History,
    /// Name of the loaded model or project, used to name exported files.
    model_name: String,
    /// Where textures were last exported, or the loaded file's folder.
    export_directory: Option<PathBuf>,
    project_path: Option<PathBuf>,
    /// `history.state()` when the project was last saved or loaded.
    saved_state: u64,
    window_title: String,
}

impl RustPaintApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let document = Document::new(mesh::primitives::cube());
        renderer::viewport::init(cc, &document);
        Self {
            brush: Brush::default(),
            camera: OrbitCamera::default(),
            status: "Use File > Open model... to load a .glb / .gltf file.".to_string(),
            selected_layer: document.layers[document.layers.len() - 1].id,
            document,
            layers_panel: LayersPanelState::default(),
            thumbnails: HashMap::new(),
            revision: 0,
            thumbnails_revision: None,
            opacity_drag: None,
            painter: None,
            stroke: None,
            history: History::default(),
            model_name: "cube".to_string(),
            export_directory: None,
            project_path: None,
            saved_state: 0,
            window_title: String::new(),
        }
    }

    /// Replaces the open document everywhere.
    fn set_document(&mut self, document: Document, frame: &eframe::Frame) {
        if let Some(render_state) = frame.wgpu_render_state() {
            renderer::viewport::show_document(render_state, &document);
        }
        let (center, radius) = document.mesh.bounds();
        self.camera.frame(center, radius);
        self.selected_layer = document.layers[document.layers.len() - 1].id;
        self.document = document;
        self.thumbnails.clear();
        self.revision += 1;
        self.opacity_drag = None;
        self.painter = None;
        self.stroke = None;
        self.history.clear();
        self.saved_state = self.history.state();
    }

    fn open(&mut self, path: &Path, frame: &eframe::Frame) {
        let is_project = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case(PROJECT_EXTENSION));
        if is_project {
            self.load_project(path, frame);
        } else {
            self.load_model(path, frame);
        }
    }

    fn load_model(&mut self, path: &Path, frame: &eframe::Frame) {
        let name = file_name(path);
        match mesh::gltf_import::load(path) {
            Ok(mut mesh) => {
                give_solid_materials_a_canvas(&mut mesh);
                self.status = format!(
                    "Loaded {name}: {} vertices, {} triangles, {} materials",
                    mesh.vertices.len(),
                    mesh.indices.len() / 3,
                    mesh.materials.len()
                );
                self.set_document(Document::new(mesh), frame);
                self.remember_source(path);
                // A model isn't a project yet; the first save asks where to put it.
                self.project_path = None;
            }
            Err(error) => {
                self.status = format!("Failed to load {name}: {error}");
            }
        }
    }

    fn load_project(&mut self, path: &Path, frame: &eframe::Frame) {
        let name = file_name(path);
        match project::load(path) {
            Ok(document) => {
                self.set_document(document, frame);
                self.remember_source(path);
                self.project_path = Some(path.to_path_buf());
                self.status = format!("Opened project {name}");
            }
            Err(error) => {
                self.status = format!("Failed to open {name}: {error}");
            }
        }
    }

    /// Names exports after the opened file and suggests its folder for them.
    fn remember_source(&mut self, path: &Path) {
        if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
            self.model_name = stem.to_string();
        }
        self.export_directory = path.parent().map(Path::to_path_buf);
    }

    fn export_textures(&mut self) {
        self.finish_stroke();
        let mut dialog = rfd::FileDialog::new().set_title("Export textures to folder");
        if let Some(directory) = &self.export_directory {
            dialog = dialog.set_directory(directory);
        }
        let Some(directory) = dialog.pick_folder() else {
            return;
        };

        let files = export::plan(&directory, &self.model_name, &self.document);
        let existing: Vec<String> = files
            .iter()
            .filter(|file| file.path.exists())
            .map(|file| file_name(&file.path))
            .collect();
        if !existing.is_empty() {
            let answer = rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Warning)
                .set_title("Replace existing textures?")
                .set_description(format!(
                    "{} already exist in {}:\n\n{}\n\nReplace them?",
                    if existing.len() == 1 {
                        "This file"
                    } else {
                        "These files"
                    },
                    directory.display(),
                    existing.join("\n"),
                ))
                .set_buttons(rfd::MessageButtons::YesNo)
                .show();
            if answer != rfd::MessageDialogResult::Yes {
                self.status = "Export cancelled".to_string();
                return;
            }
        }

        match export::write(&files, &self.document) {
            Ok(()) => {
                self.status = format!(
                    "Exported {} texture{} to {}",
                    files.len(),
                    if files.len() == 1 { "" } else { "s" },
                    directory.display()
                );
                self.export_directory = Some(directory);
            }
            Err(error) => {
                self.status = format!("Export failed: {error}");
            }
        }
    }

    fn save_project(&mut self, ask_for_path: bool) {
        self.finish_stroke();
        let path = match (&self.project_path, ask_for_path) {
            (Some(path), false) => path.clone(),
            _ => {
                let picked = rfd::FileDialog::new()
                    .add_filter("RustPaint project", &[PROJECT_EXTENSION])
                    .set_file_name(format!("untitled.{PROJECT_EXTENSION}"))
                    .save_file();
                let Some(path) = picked else {
                    return;
                };
                if path.extension().is_none() {
                    path.with_extension(PROJECT_EXTENSION)
                } else {
                    path
                }
            }
        };

        let name = file_name(&path);
        match project::save(&path, &self.document) {
            Ok(()) => {
                self.project_path = Some(path);
                self.saved_state = self.history.state();
                self.status = format!("Saved {name}");
            }
            Err(error) => {
                self.status = format!("Failed to save {name}: {error}");
            }
        }
    }

    /// Makes sure `self.painter` matches the current view, rebuilding it if needed.
    fn ensure_painter(&mut self, viewport: egui::Rect) {
        if !self
            .painter
            .as_ref()
            .is_some_and(|painter| painter.matches(&self.camera, viewport))
        {
            self.painter = None;
        }
        if self.painter.is_none() {
            self.painter = Some(StrokePainter::new(
                &self.document.mesh,
                &self.camera,
                viewport,
            ));
        }
    }

    fn selected_index(&self) -> usize {
        self.document
            .layer_index(self.selected_layer)
            .unwrap_or(self.document.layers.len() - 1)
    }

    fn paint(
        &mut self,
        viewport: egui::Rect,
        from: egui::Pos2,
        to: egui::Pos2,
        frame: &eframe::Frame,
    ) {
        self.ensure_painter(viewport);
        let Some(painter) = self.painter.as_ref() else {
            return;
        };
        let index = self.selected_index();
        if self.stroke.is_none() {
            self.stroke = Stroke::begin(&self.document, index);
            if self.stroke.is_none() {
                self.status = "The base layer can't be painted on. Select or add a layer.".into();
                return;
            }
        }
        let Some(stroke) = self.stroke.as_mut() else {
            return;
        };

        let sizes = self.document.slot_sizes();
        let layer = &mut self.document.layers[index];
        let id = layer.id;
        let changed = painter.paint_segment(stroke, layer, &sizes, from, to, &self.brush);
        let dirty: Vec<Dirty> = changed
            .into_iter()
            .map(|(material, region)| Dirty::Pixels {
                layer: id,
                material,
                region,
            })
            .collect();
        self.show_changes(&dirty, frame);
    }

    /// Ends the current stroke and records it for undo.
    fn finish_stroke(&mut self) {
        if let Some(stroke) = self.stroke.take() {
            self.history.push(stroke.finish());
        }
    }

    fn step_history(&mut self, frame: &eframe::Frame, redo: bool) {
        self.finish_stroke();
        self.opacity_drag = None;
        let dirty = if redo {
            self.history.redo(&mut self.document)
        } else {
            self.history.undo(&mut self.document)
        };
        self.show_changes(&dirty, frame);
    }

    /// Applies `change`, shows it, and records it for undo.
    fn commit(&mut self, mut change: Change, frame: &eframe::Frame) {
        if let Some(dirty) = self.document.apply(&mut change) {
            self.history.push(vec![change]);
            self.show_changes(&[dirty], frame);
        }
    }

    /// Updates the viewport after document changes and marks thumbnails stale.
    fn show_changes(&mut self, dirty: &[Dirty], frame: &eframe::Frame) {
        if dirty.is_empty() {
            return;
        }
        self.revision += 1;
        if let Some(render_state) = frame.wgpu_render_state() {
            renderer::viewport::apply_changes(render_state, &self.document, dirty);
        }
    }

    fn handle_layer_action(&mut self, action: LayerAction, frame: &eframe::Frame) {
        self.finish_stroke();
        let props_change = |layer: usize, document: &Document, edit: &dyn Fn(&mut LayerProps)| {
            let layer = document.layers.get(layer)?;
            let mut props = layer.props();
            edit(&mut props);
            (props != layer.props()).then_some(Change::Props {
                layer: layer.id,
                props,
            })
        };
        match action {
            LayerAction::Select(index) => {
                if let Some(layer) = self.document.layers.get(index) {
                    self.selected_layer = layer.id;
                }
            }
            LayerAction::Add => {
                let index = self.selected_index() + 1;
                let change = self.document.insert_new_layer(index);
                self.history.push(vec![change]);
                self.selected_layer =
                    self.document.layers[index.min(self.document.layers.len() - 1)].id;
                self.show_changes(&[Dirty::Layers], frame);
            }
            LayerAction::Delete(index) => {
                // The base layer (index 0) can't be deleted.
                let Some(below) = index.checked_sub(1) else {
                    return;
                };
                let below = self.document.layers.get(below).map(|layer| layer.id);
                self.commit(
                    Change::Layer {
                        index,
                        stored: None,
                    },
                    frame,
                );
                if let Some(id) = below {
                    self.selected_layer = id;
                }
            }
            LayerAction::Move { from, to } => self.commit(Change::Move { from, to }, frame),
            LayerAction::SetVisible(index, visible) => {
                if let Some(change) = props_change(index, &self.document, &|p| p.visible = visible)
                {
                    self.commit(change, frame);
                }
            }
            LayerAction::SetTarget { layer, slot, on } => {
                let edit = |p: &mut LayerProps| {
                    if let Some(target) = p.targets.get_mut(slot) {
                        *target = on;
                    }
                };
                if let Some(change) = props_change(layer, &self.document, &edit) {
                    self.commit(change, frame);
                }
            }
            LayerAction::Rename(index, name) => {
                if let Some(change) =
                    props_change(index, &self.document, &|p| p.name = name.clone())
                {
                    self.commit(change, frame);
                }
            }
            LayerAction::Opacity { layer, value, done } => {
                let Some(current) = self.document.layers.get_mut(layer) else {
                    return;
                };
                // Show the new opacity live; record one undo step when the drag ends.
                let before = match self.opacity_drag.take() {
                    Some((id, props)) if id == current.id => props,
                    _ => current.props(),
                };
                current.opacity = value;
                let id = current.id;
                self.show_changes(&[Dirty::Layers], frame);
                if done {
                    if before.opacity != value {
                        self.history.push(vec![Change::Props {
                            layer: id,
                            props: before,
                        }]);
                    }
                } else {
                    self.opacity_drag = Some((id, before));
                }
            }
        }
    }

    /// Regenerates layer thumbnails after the document changed.
    fn update_thumbnails(&mut self, ctx: &egui::Context) {
        if self.thumbnails_revision == Some(self.revision) {
            return;
        }
        self.thumbnails_revision = Some(self.revision);
        let mut fresh = HashMap::new();
        for index in 0..self.document.layers.len() {
            let id = self.document.layers[index].id;
            let image = ui::layers_panel::thumbnail(&self.document, index);
            let options = egui::TextureOptions::LINEAR;
            let handle = match self.thumbnails.remove(&id) {
                Some(mut handle) => {
                    handle.set(image, options);
                    handle
                }
                None => ctx.load_texture(format!("layer-thumbnail-{id}"), image, options),
            };
            fresh.insert(id, handle);
        }
        self.thumbnails = fresh;
    }

    fn has_unsaved_changes(&self) -> bool {
        self.history.state() != self.saved_state
            || self.stroke.as_ref().is_some_and(Stroke::has_changes)
    }

    /// The surface brush shows as a disc lying on the model; otherwise a screen circle.
    fn brush_cursor(
        &mut self,
        viewport: egui::Rect,
        pointer: egui::Pos2,
        camera_moved: bool,
    ) -> BrushCursor {
        let circle = BrushCursor::Circle(pointer, self.brush.size * 0.5);
        // While the camera moves, skip rebuilding the painter every frame.
        if self.brush.screen_space || camera_moved {
            return circle;
        }
        self.ensure_painter(viewport);
        self.painter
            .as_ref()
            .and_then(|painter| painter.brush_outline(pointer, &self.brush))
            .map_or(circle, BrushCursor::Outline)
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        let document = self
            .project_path
            .as_deref()
            .map_or_else(|| "Untitled".to_string(), file_name);
        let marker = if self.has_unsaved_changes() { "*" } else { "" };
        let title = format!("{document}{marker} - RustPaint");
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
    }
}

impl eframe::App for RustPaintApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let menu_state = MenuState {
            can_undo: self.history.can_undo() || self.stroke.is_some(),
            can_redo: self.history.can_redo(),
        };
        match ui::menu_bar::show(ui, &menu_state) {
            Some(MenuAction::OpenModel) => {
                let picked = rfd::FileDialog::new()
                    .add_filter("glTF", &["glb", "gltf"])
                    .pick_file();
                if let Some(path) = picked {
                    self.load_model(&path, frame);
                }
            }
            Some(MenuAction::OpenProject) => {
                let picked = rfd::FileDialog::new()
                    .add_filter("RustPaint project", &[PROJECT_EXTENSION])
                    .pick_file();
                if let Some(path) = picked {
                    self.load_project(&path, frame);
                }
            }
            Some(MenuAction::SaveProject) => self.save_project(false),
            Some(MenuAction::SaveProjectAs) => self.save_project(true),
            Some(MenuAction::ExportTextures) => self.export_textures(),
            Some(MenuAction::Undo) => self.step_history(frame, false),
            Some(MenuAction::Redo) => self.step_history(frame, true),
            None => {}
        }

        // Drag and drop (works on some platforms only, for now)
        let dropped = ui.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(path) = dropped {
            self.open(&path, frame);
        }

        // Outer panels first, central panel last.
        ui::status_bar::show(ui, &self.status);
        ui::tools_panel::show(ui, &mut self.brush);
        self.update_thumbnails(ui.ctx());
        let selected = self.selected_index();
        let actions = ui::layers_panel::show(
            ui,
            &self.document,
            selected,
            &self.thumbnails,
            &mut self.layers_panel,
        );
        for action in actions {
            self.handle_layer_action(action, frame);
        }
        let input = ui::viewport::show(ui, &mut self.camera);
        if let Some((from, to)) = input.paint {
            self.paint(input.rect, from, to, frame);
        }
        if input.stroke_finished {
            self.finish_stroke();
        }
        if let Some(pointer) = input.cursor {
            let cursor = self.brush_cursor(input.rect, pointer, input.camera_moved);
            ui::viewport::draw_brush_cursor(ui, input.rect, cursor);
        }

        self.update_window_title(ui.ctx());
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string()
}

/// glTF materials without a base colour texture import as a 1x1 image,
/// which has no room to paint. Replace those with a full-size canvas.
fn give_solid_materials_a_canvas(mesh: &mut Mesh) {
    for material in &mut mesh.materials {
        let image = &material.base_color;
        if image.width == 1 && image.height == 1 {
            let pixel = [image.rgba[0], image.rgba[1], image.rgba[2], image.rgba[3]];
            material.base_color = Image::filled(DEFAULT_CANVAS_SIZE, DEFAULT_CANVAS_SIZE, pixel);
        }
    }
}
