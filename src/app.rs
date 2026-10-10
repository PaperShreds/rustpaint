use crate::mesh;
use crate::painting::brush::Brush;
use crate::renderer;
use crate::renderer::camera::OrbitCamera;
use crate::ui;
use crate::ui::menu_bar::MenuAction;
use eframe::egui;
use std::path::Path;

pub struct RustPaintApp {
    pub brush: Brush,
    pub layers: Vec<String>,
    pub selected_layer: usize,
    pub material_names: Vec<String>,
    pub selected_material: usize,
    pub camera: OrbitCamera,
    pub status: String,
}

impl Default for RustPaintApp {
    fn default() -> Self {
        Self {
            brush: Brush::default(),
            layers: vec!["Layer 1".to_string()],
            selected_layer: 0,
            material_names: vec!["Checker".to_string()],
            selected_material: 0,
            camera: OrbitCamera::default(),
            status: "Use File > Open model... to load a .glb / .gltf file.".to_string(),
        }
    }
}

impl RustPaintApp {
    fn load_model(&mut self, path: &Path, frame: &eframe::Frame) {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model");

        match mesh::gltf_import::load(path) {
            Ok(mesh) => {
                if let Some(render_state) = frame.wgpu_render_state() {
                    renderer::viewport::upload_mesh(render_state, &mesh);
                }
                let (center, radius) = mesh.bounds();
                self.camera.frame(center, radius);

                self.material_names = mesh.materials.iter().map(|m| m.name.clone()).collect();
                self.selected_material = 0;

                self.status = format!(
                    "Loaded {name}: {} vertices, {} triangles, {} materials",
                    mesh.vertices.len(),
                    mesh.indices.len() / 3,
                    mesh.materials.len()
                );
            }
            Err(error) => {
                self.status = format!("Failed to load {name}: {error}");
            }
        }
    }
}

impl eframe::App for RustPaintApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // File > Open model...
        if let Some(MenuAction::OpenModel) = ui::menu_bar::show(ui) {
            let picked = rfd::FileDialog::new()
                .add_filter("glTF", &["glb", "gltf"])
                .pick_file();
            if let Some(path) = picked {
                self.load_model(&path, frame);
            }
        }

        // Drag and drop (works on some platforms only, for now)
        let dropped = ui.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(path) = dropped {
            self.load_model(&path, frame);
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
        ui::viewport::show(ui, &mut self.camera);
    }
}