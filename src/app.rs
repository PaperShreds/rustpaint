use crate::document::{Change, Dirty, Document, LayerProps};
use crate::export;
use crate::mesh::{self, Image, Mesh};
use crate::painting::brush::{self, Brush, FillMode, Tool};
use crate::painting::history::History;
use crate::painting::stroke::{Stroke, StrokePainter};
use crate::project;
use crate::renderer;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::environment::{EnvironmentSource, HdrImage};
use crate::renderer::viewport::{EnvironmentSettings, ViewMode};
use crate::settings::{self, Settings};
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

/// An action waiting for the user to decide what happens to unsaved changes.
enum Pending {
    Quit,
    Open(PathBuf),
}

pub struct RustPaintApp {
    pub brush: Brush,
    pub camera: OrbitCamera,
    /// Camera distance last frame, to scale the brush with zooming.
    zoom_distance: f32,
    pub view_mode: ViewMode,
    pub environment: EnvironmentSettings,
    environment_source: EnvironmentSource,
    /// File > Open recent, most recent first.
    recent_files: Vec<PathBuf>,
    pub status: String,
    /// Shown as the "Save changes?" dialog while set.
    pending: Option<Pending>,
    /// Set once the user agreed to close, so the next close request goes through.
    allow_close: bool,
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
        let settings = Settings::load(cc.storage);
        let mut status = "Use File > Open model... to load a .glb / .gltf file.".to_string();
        let mut environment_source = settings.environment_source;
        // A custom HDRI may have been moved or deleted since: fall back to the default.
        let environment = environment_source.load().or_else(|error| {
            status = format!("Could not load {}: {error}", environment_source.name());
            environment_source = EnvironmentSource::default();
            environment_source.load()
        });
        let environment = environment.unwrap_or_else(|error| {
            eprintln!("Could not load the built-in environment: {error}");
            HdrImage {
                width: 1,
                height: 1,
                rgba: vec![0.5, 0.5, 0.5, 1.0],
            }
        });
        let document = Document::new(mesh::primitives::cube());
        renderer::viewport::init(cc, &document, &environment);
        Self {
            brush: settings.brush,
            camera: OrbitCamera::default(),
            zoom_distance: OrbitCamera::default().distance,
            view_mode: settings.view_mode,
            environment: settings.environment,
            environment_source,
            recent_files: settings.recent_files,
            pending: None,
            allow_close: false,
            status,
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
        // Reframing for a new model isn't a zoom: leave the brush alone.
        self.zoom_distance = self.camera.distance;
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
        if !path.exists() {
            self.status = format!("{} no longer exists", path.display());
            self.recent_files.retain(|recent| recent != path);
            return;
        }
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
                settings::add_recent_file(&mut self.recent_files, path);
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
                settings::add_recent_file(&mut self.recent_files, path);
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
                settings::add_recent_file(&mut self.recent_files, &path);
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
        } else if self.brush.tool == Tool::Fill {
            // One fill per click: the rest of the drag does nothing.
            return;
        }
        let Some(stroke) = self.stroke.as_mut() else {
            return;
        };

        let sizes = self.document.slot_sizes();
        let layer = &mut self.document.layers[index];
        let id = layer.id;
        let changed = if self.brush.tool == Tool::Fill {
            let changed = painter.fill(stroke, layer, &sizes, to, &self.brush);
            if changed.is_empty() {
                self.status = "Nothing to fill there: click the model where it uses a \
                               material this layer paints on."
                    .into();
            }
            changed
        } else {
            painter.paint_segment(stroke, layer, &sizes, from, to, &self.brush)
        };
        let dirty: Vec<Dirty> = changed
            .into_iter()
            .map(|(material, channel, region)| Dirty::Pixels {
                layer: id,
                material,
                channel,
                region,
            })
            .collect();
        self.show_changes(&dirty, frame);
    }

    /// Ends the current stroke and records it for undo.
    fn finish_stroke(&mut self) {
        if let Some(stroke) = self.stroke.take() {
            let label = self.stroke_label(stroke.layer());
            self.history.push(label, stroke.finish());
        }
    }

    /// The history panel's name for a stroke on layer `id` with the current tool.
    fn stroke_label(&self, id: u64) -> String {
        let name = self.layer_name(id);
        match self.brush.tool {
            Tool::Paint => format!("Paint on {name}"),
            Tool::Erase => format!("Erase on {name}"),
            Tool::Fill => match self.brush.fill_mode {
                FillMode::Layer => format!("Fill {name}"),
                FillMode::Material => format!("Fill material on {name}"),
                FillMode::UvIsland => format!("Fill UV island on {name}"),
            },
        }
    }

    /// A layer's name, quoted for messages ("?" if it no longer exists).
    fn layer_name(&self, id: u64) -> String {
        self.document.layer_index(id).map_or_else(
            || "?".to_string(),
            |index| self.document.layers[index].name.clone(),
        )
    }

    /// Name of the layer at `index` (empty if there is none).
    fn name_at(&self, index: usize) -> String {
        self.document
            .layers
            .get(index)
            .map_or_else(String::new, |layer| layer.name.clone())
    }

    /// Jumps to a step picked in the history panel.
    fn go_to_history(&mut self, position: usize, frame: &eframe::Frame) {
        self.finish_stroke();
        self.opacity_drag = None;
        let dirty = self.history.go_to(&mut self.document, position);
        self.show_changes(&dirty, frame);
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

    /// Applies `change`, shows it, and records it for undo as `label`.
    fn commit(&mut self, label: String, mut change: Change, frame: &eframe::Frame) {
        if let Some(dirty) = self.document.apply(&mut change) {
            self.history.push(label, vec![change]);
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
                let added = self.name_at(index);
                self.history.push(format!("Add {added}"), vec![change]);
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
                    format!("Delete {}", self.name_at(index)),
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
            LayerAction::Move { from, to } => {
                let direction = if to > from { "up" } else { "down" };
                let label = format!("Move {} {direction}", self.name_at(from));
                self.commit(label, Change::Move { from, to }, frame)
            }
            LayerAction::SetVisible(index, visible) => {
                if let Some(change) = props_change(index, &self.document, &|p| p.visible = visible)
                {
                    let verb = if visible { "Show" } else { "Hide" };
                    self.commit(format!("{verb} {}", self.name_at(index)), change, frame);
                }
            }
            LayerAction::SetTarget { layer, slot, on } => {
                let edit = |p: &mut LayerProps| {
                    if let Some(target) = p.targets.get_mut(slot) {
                        *target = on;
                    }
                };
                if let Some(change) = props_change(layer, &self.document, &edit) {
                    self.commit(
                        format!("Change materials of {}", self.name_at(layer)),
                        change,
                        frame,
                    );
                }
            }
            LayerAction::SetBlend(index, blend) => {
                if let Some(change) = props_change(index, &self.document, &|p| p.blend = blend) {
                    let label = format!("Blend {} as {}", self.name_at(index), blend.name());
                    self.commit(label, change, frame);
                }
            }
            LayerAction::Rename(index, new_name) => {
                if let Some(change) =
                    props_change(index, &self.document, &|p| p.name = new_name.clone())
                {
                    let label = format!("Rename {} to {new_name}", self.name_at(index));
                    self.commit(label, change, frame);
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
                        let label = format!(
                            "Opacity of {} to {:.0}%",
                            self.layer_name(id),
                            value * 100.0
                        );
                        self.history.push(
                            label,
                            vec![Change::Props {
                                layer: id,
                                props: before,
                            }],
                        );
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

    /// With "Keep size on model", zooming in makes the brush bigger on screen
    /// (and zooming out smaller), so it keeps covering the same area.
    fn follow_zoom(&mut self) {
        let distance = self.camera.distance;
        if self.brush.keep_size_on_model && distance != self.zoom_distance {
            self.brush.size = (self.brush.size * self.zoom_distance / distance)
                .clamp(brush::MIN_SIZE, brush::MAX_SIZE);
        }
        self.zoom_distance = distance;
    }

    /// Opens a file, first asking about unsaved changes if there are any.
    fn request_open(&mut self, path: PathBuf, frame: &eframe::Frame) {
        if self.has_unsaved_changes() {
            self.pending = Some(Pending::Open(path));
        } else {
            self.open(&path, frame);
        }
    }

    fn proceed(&mut self, pending: Pending, ctx: &egui::Context, frame: &eframe::Frame) {
        match pending {
            Pending::Quit => {
                self.allow_close = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            Pending::Open(path) => self.open(&path, frame),
        }
    }

    /// "Save changes before ...?" with Save / Don't save / Cancel.
    fn unsaved_changes_dialog(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let Some(pending) = &self.pending else {
            return;
        };
        let verb = match pending {
            Pending::Quit => "closing",
            Pending::Open(_) => "opening another file",
        };
        let document = self
            .project_path
            .as_deref()
            .map_or_else(|| "Untitled".to_string(), file_name);

        #[derive(Clone, Copy)]
        enum Choice {
            Save,
            Discard,
            Cancel,
        }
        let mut choice = None;
        let modal = egui::Modal::new(egui::Id::new("unsaved_changes")).show(ctx, |ui| {
            ui.heading("Unsaved changes");
            ui.label(format!("Save changes to {document} before {verb}?"));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    choice = Some(Choice::Save);
                }
                if ui.button("Don't save").clicked() {
                    choice = Some(Choice::Discard);
                }
                if ui.button("Cancel").clicked() {
                    choice = Some(Choice::Cancel);
                }
            });
        });
        // Escape or a click outside counts as Cancel.
        if modal.should_close() && choice.is_none() {
            choice = Some(Choice::Cancel);
        }
        // Only take `pending` once a choice is made, or the dialog closes itself.
        let Some(choice) = choice else {
            return;
        };
        let Some(pending) = self.pending.take() else {
            return;
        };
        match choice {
            Choice::Save => {
                self.save_project(false);
                // Only go ahead if the save actually happened (not cancelled or failed).
                if !self.has_unsaved_changes() {
                    self.proceed(pending, ctx, frame);
                }
            }
            Choice::Discard => self.proceed(pending, ctx, frame),
            Choice::Cancel => {}
        }
    }

    fn load_environment(&mut self, frame: &eframe::Frame) {
        let picked = rfd::FileDialog::new()
            .add_filter("HDR image", &["hdr", "exr"])
            .pick_file();
        if let Some(path) = picked {
            self.set_environment(EnvironmentSource::File(path), frame);
        }
    }

    fn set_environment(&mut self, source: EnvironmentSource, frame: &eframe::Frame) {
        let name = source.name();
        match source.load() {
            Ok(image) => {
                if let Some(render_state) = frame.wgpu_render_state() {
                    renderer::viewport::set_environment(render_state, &image);
                }
                self.status = format!("Lighting with {name} ({}x{})", image.width, image.height);
                self.environment_source = source;
            }
            Err(error) => self.status = format!("Failed to load {name}: {error}"),
        }
    }

    fn settings(&self) -> Settings {
        Settings {
            recent_files: self.recent_files.clone(),
            environment_source: self.environment_source.clone(),
            environment: self.environment,
            view_mode: self.view_mode,
            // Always start with the paint brush.
            brush: Brush {
                tool: Tool::Paint,
                ..self.brush.clone()
            },
        }
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
    /// Called by eframe every so often and on exit.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.settings().save(storage);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let menu_state = MenuState {
            can_undo: self.history.can_undo() || self.stroke.is_some(),
            can_redo: self.history.can_redo(),
            view_mode: self.view_mode,
            recent_files: &self.recent_files,
            environment_source: &self.environment_source,
        };
        // Closing with unsaved work: stop, and ask first.
        let close_requested = ui.input(|i| i.viewport().close_requested());
        if close_requested && !self.allow_close && self.has_unsaved_changes() {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending = Some(Pending::Quit);
        }

        match ui::menu_bar::show(ui, &menu_state, &mut self.environment) {
            Some(MenuAction::OpenModel) => {
                let picked = rfd::FileDialog::new()
                    .add_filter("glTF", &["glb", "gltf"])
                    .pick_file();
                if let Some(path) = picked {
                    self.request_open(path, frame);
                }
            }
            Some(MenuAction::OpenProject) => {
                let picked = rfd::FileDialog::new()
                    .add_filter("RustPaint project", &[PROJECT_EXTENSION])
                    .pick_file();
                if let Some(path) = picked {
                    self.request_open(path, frame);
                }
            }
            Some(MenuAction::OpenRecent(path)) => self.request_open(path, frame),
            Some(MenuAction::ClearRecent) => self.recent_files.clear(),
            Some(MenuAction::LoadEnvironment) => self.load_environment(frame),
            Some(MenuAction::SetEnvironment(built_in)) => {
                self.set_environment(EnvironmentSource::BuiltIn(built_in), frame)
            }
            Some(MenuAction::SaveProject) => self.save_project(false),
            Some(MenuAction::SaveProjectAs) => self.save_project(true),
            Some(MenuAction::ExportTextures) => self.export_textures(),
            Some(MenuAction::Undo) => self.step_history(frame, false),
            Some(MenuAction::Redo) => self.step_history(frame, true),
            Some(MenuAction::SetView(mode)) => self.view_mode = mode,
            None => {}
        }

        // Drag and drop (works on some platforms only, for now)
        let dropped = ui.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(path) = dropped {
            self.request_open(path, frame);
        }

        // Outer panels first, central panel last.
        ui::status_bar::show(ui, &self.status);
        ui::tools_panel::show(ui, &mut self.brush);
        self.update_thumbnails(ui.ctx());
        let selected = self.selected_index();
        // Right column: layers, with the undo history docked below them.
        let (actions, history_pick) = egui::Panel::right("layers_panel")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                let history_pick = ui::history_panel::show(ui, &self.history);
                let actions = ui::layers_panel::show(
                    ui,
                    &self.document,
                    selected,
                    &self.thumbnails,
                    &mut self.layers_panel,
                );
                (actions, history_pick)
            })
            .inner;
        for action in actions {
            self.handle_layer_action(action, frame);
        }
        if let Some(position) = history_pick {
            self.go_to_history(position, frame);
        }
        let input = ui::viewport::show(ui, &mut self.camera, self.view_mode, self.environment);
        self.follow_zoom();
        if let Some((from, to)) = input.paint {
            self.paint(input.rect, from, to, frame);
        }
        if input.stroke_finished {
            self.finish_stroke();
        }
        if let Some(pointer) = input.cursor {
            if self.brush.tool == Tool::Fill {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            } else {
                let cursor = self.brush_cursor(input.rect, pointer, input.camera_moved);
                ui::viewport::draw_brush_cursor(ui, input.rect, cursor);
            }
        }

        let ctx = ui.ctx().clone();
        self.unsaved_changes_dialog(&ctx, frame);
        self.update_window_title(&ctx);
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
