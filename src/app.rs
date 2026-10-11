use crate::document::{Change, Dirty, Document, LayerProps};
use crate::export;
use crate::mesh::{self, Image, Mesh};
use crate::painting::brush::{self, Brush, FillMode, Tool};
use crate::painting::history::History;
use crate::painting::stroke::{Stroke, StrokePainter};
use crate::project;
use crate::recovery::{self, Recovered, Session};
use crate::renderer;
use crate::renderer::camera::OrbitCamera;
use crate::renderer::environment::{EnvironmentSource, HdrImage};
use crate::renderer::viewport::{EnvironmentSettings, ViewMode};
use crate::settings::{self, Autosave, Settings};
use crate::ui;
use crate::ui::layers_panel::{LayerAction, LayersPanelState};
use crate::ui::menu_bar::{MenuAction, MenuState};
use crate::ui::viewport::BrushCursor;
use eframe::egui;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

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
    thumbnails: ui::layers_panel::Thumbnails,
    /// The brush paints the selected layer's mask instead of its channels
    /// (when it has one).
    editing_mask: bool,
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
    /// This instance's autosave slot; `None` if autosave couldn't start.
    session: Option<Session>,
    /// Autosaves left behind by a crash, offered for restoring one at a time.
    recovery_offers: Vec<Recovered>,
    last_autosave: Instant,
    autosave_options: Autosave,
    /// `history.state()` of the current recovery copy, if there is one.
    autosaved_state: Option<u64>,
}

impl RustPaintApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Lets egui draw the SVG tool icons.
        egui_extras::install_image_loaders(&cc.egui_ctx);
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
        let (session, recovery_offers) = match eframe::storage_dir("RustPaint") {
            Some(directory) => match Session::start(&directory.join("recovery")) {
                Ok((session, offers)) => (Some(session), offers),
                Err(error) => {
                    status = format!("Autosave is off: {error}");
                    (None, Vec::new())
                }
            },
            None => (None, Vec::new()),
        };
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
            editing_mask: false,
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
            session,
            recovery_offers,
            last_autosave: Instant::now(),
            autosave_options: settings.autosave,
            autosaved_state: None,
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
        self.editing_mask = false;
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

    fn export_model(&mut self) {
        self.finish_stroke();
        let mut dialog = rfd::FileDialog::new()
            .set_title("Export model")
            .add_filter("glTF binary", &["glb"])
            .set_file_name(format!("{}.glb", self.model_name));
        if let Some(directory) = &self.export_directory {
            dialog = dialog.set_directory(directory);
        }
        let Some(mut path) = dialog.save_file() else {
            return;
        };
        if path.extension().is_none() {
            path.set_extension("glb");
        }
        let name = file_name(&path);
        match export::glb::write(&path, &self.document, &self.model_name) {
            Ok(()) => {
                self.status = format!("Exported {name}");
                self.export_directory = path.parent().map(Path::to_path_buf);
            }
            Err(error) => self.status = format!("Failed to export {name}: {error}"),
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
            self.stroke = if self.painting_mask() {
                Stroke::begin_on_mask(&self.document, index)
            } else {
                Stroke::begin(&self.document, index)
            };
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
            .map(|(material, plane, region)| Dirty::Pixels {
                layer: id,
                material,
                plane,
                region,
            })
            .collect();
        self.show_changes(&dirty, frame);
    }

    /// Ends the current stroke and records it for undo.
    fn finish_stroke(&mut self) {
        if let Some(stroke) = self.stroke.take() {
            let label = self.stroke_label(stroke.layer(), stroke.on_mask());
            self.history.push(label, stroke.finish());
        }
    }

    /// Whether the brush paints the selected layer's mask.
    fn painting_mask(&self) -> bool {
        self.editing_mask && self.document.layers[self.selected_index()].mask.is_some()
    }

    /// The history panel's name for a stroke on layer `id` (on its mask, if
    /// `on_mask`) with the current tool.
    fn stroke_label(&self, id: u64, on_mask: bool) -> String {
        let name = self.layer_name(id);
        let name = if on_mask {
            format!("{name} mask")
        } else {
            name
        };
        let erase = self.brush.erases();
        // On a mask, painting reveals the layer and erasing hides it.
        match self.brush.tool {
            Tool::Paint if on_mask => {
                let verb = if erase { "Hide" } else { "Reveal" };
                format!("{verb} on {name}")
            }
            Tool::Paint if erase => format!("Erase on {name}"),
            Tool::Paint => format!("Paint on {name}"),
            Tool::Smear => format!("Smear on {name}"),
            Tool::Blur => format!("Blur on {name}"),
            Tool::Fill => {
                let verb = match (on_mask, erase) {
                    (true, true) => "Hide",
                    (true, false) => "Reveal",
                    (false, true) => "Erase",
                    (false, false) => "Fill",
                };
                match self.brush.fill_mode {
                    FillMode::Layer => format!("{verb} all of {name}"),
                    FillMode::Material => format!("{verb} material on {name}"),
                    FillMode::UvIsland => format!("{verb} UV island on {name}"),
                }
            }
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
                    self.editing_mask = false;
                }
            }
            LayerAction::EditMask(index, on_mask) => {
                if let Some(layer) = self.document.layers.get(index) {
                    self.selected_layer = layer.id;
                    self.editing_mask = on_mask;
                }
            }
            LayerAction::AddMask { layer, hide } => {
                let value = if hide { 0 } else { 255 };
                if let Some(change) = self.document.set_mask(layer, Some(value)) {
                    let id = self.document.layers[layer].id;
                    self.history
                        .push(format!("Add mask to {}", self.name_at(layer)), vec![change]);
                    self.selected_layer = id;
                    // Ready to paint it.
                    self.editing_mask = true;
                    self.show_changes(&[Dirty::Mask { layer: id }], frame);
                }
            }
            LayerAction::DeleteMask(layer) => {
                if let Some(change) = self.document.set_mask(layer, None) {
                    let id = self.document.layers[layer].id;
                    let label = format!("Delete mask of {}", self.name_at(layer));
                    self.history.push(label, vec![change]);
                    self.editing_mask = false;
                    self.show_changes(&[Dirty::Mask { layer: id }], frame);
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
            let paint = Some(ui::layers_panel::thumbnail(&self.document, index));
            let mask = ui::layers_panel::mask_thumbnail(&self.document, index);
            for (is_mask, image) in [(false, paint), (true, mask)] {
                let Some(image) = image else {
                    continue;
                };
                let options = egui::TextureOptions::LINEAR;
                let handle = match self.thumbnails.remove(&(id, is_mask)) {
                    Some(mut handle) => {
                        handle.set(image, options);
                        handle
                    }
                    None => {
                        let name = format!("layer-thumbnail-{id}-{is_mask}");
                        ctx.load_texture(name, image, options)
                    }
                };
                fresh.insert((id, is_mask), handle);
            }
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
                erase: false,
                ..self.brush.clone()
            },
            autosave: self.autosave_options,
        }
    }

    /// Writes a recovery copy every so often (`autosave_options`) while there
    /// are unsaved changes (between strokes), and removes it once there are none.
    fn autosave(&mut self, ctx: &egui::Context) {
        let unsaved = self.has_unsaved_changes();
        let Some(session) = self.session.as_mut() else {
            return;
        };
        if !unsaved {
            if self.autosaved_state.take().is_some() {
                session.clear();
            }
            return;
        }
        let state = self.history.state();
        if !self.autosave_options.enabled
            || self.autosaved_state == Some(state)
            || self.stroke.is_some()
        {
            return;
        }
        let interval = self.autosave_options.interval();
        let remaining = interval.saturating_sub(self.last_autosave.elapsed());
        if !remaining.is_zero() {
            ctx.request_repaint_after(remaining);
            return;
        }
        let info = recovery::Info {
            project_path: self.project_path.clone(),
            model_name: self.model_name.clone(),
        };
        match session.autosave(&self.document, &info) {
            Ok(true) => {
                self.autosaved_state = Some(state);
                self.last_autosave = Instant::now();
            }
            // The previous autosave is still being written; try again shortly.
            Ok(false) => ctx.request_repaint_after(Duration::from_millis(250)),
            Err(error) => {
                self.status = format!("Autosave failed: {error}");
                self.last_autosave = Instant::now();
            }
        }
    }

    /// After a crash: "Recover unsaved work?" with Restore / Discard / Not now.
    fn recovery_dialog(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        if self.pending.is_some() {
            return;
        }
        let Some(offer) = self.recovery_offers.first() else {
            return;
        };
        let name = offer.info.project_path.as_deref().map_or_else(
            || format!("{} (never saved as a project)", offer.info.model_name),
            file_name,
        );
        let age = describe_age(offer.saved_at);

        #[derive(Clone, Copy)]
        enum Choice {
            Restore,
            Discard,
            Later,
        }
        let mut choice = None;
        let modal = egui::Modal::new(egui::Id::new("recover_work")).show(ctx, |ui| {
            ui.heading("Recover unsaved work?");
            ui.label("RustPaint didn't close properly last time.");
            ui.label(format!("An autosave of {name} from {age} was found."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Restore").clicked() {
                    choice = Some(Choice::Restore);
                }
                if ui.button("Discard").clicked() {
                    choice = Some(Choice::Discard);
                }
                if ui
                    .button("Not now")
                    .on_hover_text("Keep the autosave and ask again next time")
                    .clicked()
                {
                    choice = Some(Choice::Later);
                }
            });
        });
        if modal.should_close() && choice.is_none() {
            choice = Some(Choice::Later);
        }
        let Some(choice) = choice else {
            return;
        };
        let offer = self.recovery_offers.remove(0);
        match choice {
            Choice::Restore => self.restore(offer, frame),
            Choice::Discard => offer.discard(),
            Choice::Later => {}
        }
    }

    fn restore(&mut self, offer: Recovered, frame: &eframe::Frame) {
        let document = match offer.load() {
            Ok(document) => document,
            Err(error) => {
                self.status = format!("Could not restore the autosave: {error}");
                return;
            }
        };
        self.set_document(document, frame);
        let info = offer.info.clone();
        if let Some(path) = &info.project_path {
            self.remember_source(path);
        }
        self.model_name = info.model_name;
        self.project_path = info.project_path;
        // Not saved anywhere yet: keep it marked as unsaved.
        self.saved_state = u64::MAX;
        if let Some(session) = self.session.as_mut() {
            match session.adopt(offer) {
                Ok(()) => self.autosaved_state = Some(self.history.state()),
                Err(error) => self.status = format!("Autosave: {error}"),
            }
        }
        self.last_autosave = Instant::now();
        self.status = "Restored the autosave. Save it to keep it.".into();
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

    /// A clean exit: the work was saved or deliberately discarded, so the
    /// recovery copy goes.
    fn on_exit(&mut self) {
        if let Some(session) = self.session.take() {
            session.close();
        }
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

        match ui::menu_bar::show(
            ui,
            &menu_state,
            &mut self.environment,
            &mut self.autosave_options,
        ) {
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
            Some(MenuAction::ExportModel) => self.export_model(),
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
        let on_mask = self.painting_mask();
        ui::tools_panel::show(ui, &mut self.brush, on_mask);
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
                    on_mask,
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
        self.recovery_dialog(&ctx, frame);
        self.autosave(&ctx);
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

/// "just now", "5 minutes ago", "3 hours ago" or "2 days ago".
fn describe_age(time: SystemTime) -> String {
    let seconds = time.elapsed().unwrap_or_default().as_secs();
    let (amount, unit) = match seconds {
        0..60 => return "just now".to_string(),
        60..3600 => (seconds / 60, "minute"),
        3600..86400 => (seconds / 3600, "hour"),
        _ => (seconds / 86400, "day"),
    };
    format!("{amount} {unit}{} ago", if amount == 1 { "" } else { "s" })
}
