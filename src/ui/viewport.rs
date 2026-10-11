use crate::renderer::camera::OrbitCamera;
use crate::renderer::viewport::{EnvironmentSettings, ViewMode, ViewportCallback};
use eframe::egui;
use eframe::egui_wgpu;
use egui::PointerButton;

/// What the user did with the brush in the viewport this frame.
pub struct ViewportInput {
    pub rect: egui::Rect,
    /// Screen-space segment to paint, in points.
    pub paint: Option<(egui::Pos2, egui::Pos2)>,
    /// The mouse button was released, ending the current stroke.
    pub stroke_finished: bool,
    /// Where to draw the brush cursor; `None` while navigating.
    pub cursor: Option<egui::Pos2>,
    /// The camera moved this frame.
    pub camera_moved: bool,
}

/// Shape of the brush cursor.
pub enum BrushCursor {
    /// A circle of this radius (points) around the pointer.
    Circle(egui::Pos2, f32),
    /// A closed outline, e.g. the brush projected onto the surface.
    Outline(Vec<egui::Pos2>),
}

/// Left drag paints. Right drag (or Alt + left drag) orbits.
/// Middle drag (or Shift + right drag) pans. Scroll zooms.
pub fn show(
    ui: &mut egui::Ui,
    camera: &mut OrbitCamera,
    view_mode: ViewMode,
    environment: EnvironmentSettings,
) -> ViewportInput {
    egui::CentralPanel::default()
        .show(ui, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let modifiers = ui.input(|i| i.modifiers);
            let navigating = modifiers.alt;
            let aspect = rect.width() / rect.height().max(1.0);
            let view_before = camera.view_proj(aspect);

            if response.dragged_by(PointerButton::Middle)
                || (response.dragged_by(PointerButton::Secondary) && modifiers.shift)
            {
                let delta = response.drag_delta();
                camera.pan(delta.x, delta.y, rect.height());
            } else if response.dragged_by(PointerButton::Secondary)
                || (response.dragged_by(PointerButton::Primary) && navigating)
            {
                let delta = response.drag_motion();
                camera.orbit(delta.x, delta.y);
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                camera.zoom(scroll);
            }

            let mut paint = None;
            if !navigating && let Some(pos) = response.interact_pointer_pos() {
                if response.clicked_by(PointerButton::Primary) {
                    paint = Some((pos, pos));
                } else if response.drag_started_by(PointerButton::Primary) {
                    // The drag only starts after the pointer moved a little; paint from where it went down.
                    let origin = ui.input(|i| i.pointer.press_origin()).unwrap_or(pos);
                    paint = Some((origin, pos));
                } else if response.dragged_by(PointerButton::Primary) {
                    let delta = response.drag_delta();
                    if delta != egui::Vec2::ZERO {
                        paint = Some((pos - delta, pos));
                    }
                }
            }
            let stroke_finished = response.drag_stopped_by(PointerButton::Primary)
                || response.clicked_by(PointerButton::Primary);

            let view = camera.view_proj(aspect);

            ui.painter()
                .rect_filled(rect, 0.0, egui::Color32::from_gray(30));
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                ViewportCallback {
                    view_proj: view,
                    eye: camera.eye().to_array(),
                    view_mode,
                    environment,
                },
            ));

            let orbiting_or_panning = ui.input(|i| {
                i.pointer.button_down(PointerButton::Secondary)
                    || i.pointer.button_down(PointerButton::Middle)
            });
            let cursor = response
                .hover_pos()
                .filter(|_| !navigating && !orbiting_or_panning);

            ViewportInput {
                rect,
                paint,
                stroke_finished,
                cursor,
                camera_moved: view != view_before,
            }
        })
        .inner
}

/// Draws the brush cursor over the viewport, outlined so it shows on light and dark surfaces.
pub fn draw_brush_cursor(ui: &egui::Ui, rect: egui::Rect, cursor: BrushCursor) {
    let painter = ui.painter_at(rect);
    let strokes = [
        egui::Stroke::new(3.0, egui::Color32::from_black_alpha(110)),
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(220)),
    ];
    for stroke in strokes {
        match &cursor {
            BrushCursor::Circle(center, radius) => {
                painter.circle_stroke(*center, *radius, stroke);
            }
            BrushCursor::Outline(points) => {
                painter.add(egui::Shape::closed_line(points.clone(), stroke));
            }
        }
    }
}
