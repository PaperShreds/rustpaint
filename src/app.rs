use crate::mesh::{self, Image, Mesh};
use crate::painting::brush::Brush;
use crate::painting::stroke::{Stroke, StrokePainter};
use crate::project;
use crate::renderer;
use crate::renderer::camera::OrbitCamera;
use crate::ui;
use crate::ui::menu_bar::MenuAction;
use crate::ui::viewport::BrushCursor;
use eframe::egui;
use std::path::{Path, PathBuf};

const PROJECT_EXTENSION: &str = "rustpaint";
/// Untextured materials get a blank canvas of this size so they can be painted.
const DEFAULT_CANVAS_SIZE: u32 = 2048;

pub struct RustPaintApp {
    pub brush: Brush,
    pub layers: Vec<String>,
    pub selected_layer: usize,
    pub material_names: Vec<String>,
    pub selected_material: usize,
    pub camera: OrbitCamera,
    pub status: String,
    /// CPU copy of the model and its textures; painting edits this, then uploads changes.
    mesh: Mesh,
    /// Screen-space acceleration structure for painting; rebuilt when the view changes.
    painter: Option<StrokePainter>,
    /// The stroke in progress while the mouse button is held.
    stroke: Option<Stroke>,
    project_path: Option<PathBuf>,
    unsaved_changes: bool,
    window_title: String,
}

impl RustPaintApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mesh = mesh::primitives::cube();
        renderer::viewport::init(cc, &mesh);
        Self {
            brush: Brush::default(),
            layers: vec!["Layer 1".to_string()],
            selected_layer: 0,
            material_names: mesh.materials.iter().map(|m| m.name.clone()).collect(),
            selected_material: 0,
            camera: OrbitCamera::default(),
            status: "Use File > Open model... to load a .glb / .gltf file.".to_string(),
            mesh,
            painter: None,
            stroke: None,
            project_path: None,
            unsaved_changes: false,
            window_title: String::new(),
        }
    }

    /// Replaces the current model (and its textures) everywhere.
    fn set_mesh(&mut self, mesh: Mesh, frame: &eframe::Frame) {
        if let Some(render_state) = frame.wgpu_render_state() {
            renderer::viewport::upload_mesh(render_state, &mesh);
        }
        let (center, radius) = mesh.bounds();
        self.camera.frame(center, radius);
        self.material_names = mesh.materials.iter().map(|m| m.name.clone()).collect();
        self.selected_material = 0;
        self.mesh = mesh;
        self.painter = None;
        self.stroke = None;
        self.unsaved_changes = false;
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
                self.set_mesh(mesh, frame);
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
            Ok(mesh) => {
                self.set_mesh(mesh, frame);
                self.project_path = Some(path.to_path_buf());
                self.status = format!("Opened project {name}");
            }
            Err(error) => {
                self.status = format!("Failed to open {name}: {error}");
            }
        }
    }

    fn save_project(&mut self, ask_for_path: bool) {
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
        match project::save(&path, &self.mesh) {
            Ok(()) => {
                self.project_path = Some(path);
                self.unsaved_changes = false;
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
            self.painter = Some(StrokePainter::new(&self.mesh, &self.camera, viewport));
        }
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
        if self.stroke.is_none() {
            self.stroke = Stroke::begin(&self.mesh, self.selected_material);
        }
        let Some(stroke) = self.stroke.as_mut() else {
            return;
        };

        let Some(region) = painter.paint_segment(stroke, &mut self.mesh, from, to, &self.brush)
        else {
            return;
        };
        let material = stroke.material();
        self.unsaved_changes = true;

        if let Some(render_state) = frame.wgpu_render_state() {
            renderer::viewport::update_material_texture(
                render_state,
                material,
                &self.mesh.materials[material].base_color,
                region,
            );
        }
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
        let marker = if self.unsaved_changes { "*" } else { "" };
        let title = format!("{document}{marker} - RustPaint");
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
    }
}

impl eframe::App for RustPaintApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        match ui::menu_bar::show(ui) {
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
        ui::layers_panel::show(
            ui,
            &self.material_names,
            &mut self.selected_material,
            &mut self.layers,
            &mut self.selected_layer,
        );
        let input = ui::viewport::show(ui, &mut self.camera);
        if let Some((from, to)) = input.paint {
            self.paint(input.rect, from, to, frame);
        }
        if input.stroke_finished {
            self.stroke = None;
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
