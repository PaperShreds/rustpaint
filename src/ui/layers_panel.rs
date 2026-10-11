use crate::document::{Document, Layer};
use crate::mesh::Image;
use eframe::egui;
use egui::containers::menu::{MenuButton, MenuConfig};
use std::collections::HashMap;

/// Size the thumbnail is shown at, in points.
const THUMBNAIL_POINTS: f32 = 32.0;
/// Thumbnail resolution, in pixels (sharp on high-DPI screens too).
const THUMBNAIL_PIXELS: usize = 48;

/// Something the user did in the layers panel.
pub enum LayerAction {
    Select(usize),
    Add,
    Delete(usize),
    Move {
        from: usize,
        to: usize,
    },
    SetVisible(usize, bool),
    SetTarget {
        layer: usize,
        slot: usize,
        on: bool,
    },
    Rename(usize, String),
    /// `done` is false while the slider is still being dragged.
    Opacity {
        layer: usize,
        value: f32,
        done: bool,
    },
}

/// UI state that lives between frames.
#[derive(Default)]
pub struct LayersPanelState {
    /// Layer id being renamed, and the text typed so far.
    renaming: Option<(u64, String)>,
}

pub fn show(
    ui: &mut egui::Ui,
    document: &Document,
    selected: usize,
    thumbnails: &HashMap<u64, egui::TextureHandle>,
    state: &mut LayersPanelState,
) -> Vec<LayerAction> {
    let mut actions = Vec::new();
    egui::Panel::right("layers_panel")
        .resizable(true)
        .default_size(300.0)
        .show(ui, |ui| {
            ui.heading("Layers");
            ui.separator();

            let count = document.layers.len();
            let current = &document.layers[selected];
            let editable = !current.is_base();
            ui.horizontal(|ui| {
                if ui
                    .button("+ Add")
                    .on_hover_text("Add a layer above the selected one")
                    .clicked()
                {
                    actions.push(LayerAction::Add);
                }
                if ui
                    .add_enabled(editable, egui::Button::new("Delete"))
                    .clicked()
                {
                    actions.push(LayerAction::Delete(selected));
                }
                let up = ui.add_enabled(editable && selected + 1 < count, egui::Button::new("⬆"));
                if up.on_hover_text("Move up").clicked() {
                    actions.push(LayerAction::Move {
                        from: selected,
                        to: selected + 1,
                    });
                }
                let down = ui.add_enabled(editable && selected > 1, egui::Button::new("⬇"));
                if down.on_hover_text("Move down").clicked() {
                    actions.push(LayerAction::Move {
                        from: selected,
                        to: selected - 1,
                    });
                }
            });

            let mut opacity = current.opacity;
            let slider = ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
            if slider.changed() {
                actions.push(LayerAction::Opacity {
                    layer: selected,
                    value: opacity,
                    done: !slider.dragged(),
                });
            }
            if slider.drag_stopped() {
                actions.push(LayerAction::Opacity {
                    layer: selected,
                    value: opacity,
                    done: true,
                });
            }
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("layers_grid")
                    .num_columns(4)
                    .striped(true)
                    .spacing([6.0, 4.0])
                    .show(ui, |ui| {
                        // Top layer first, like other paint programs.
                        for index in (0..count).rev() {
                            let layer = &document.layers[index];
                            layer_row(
                                ui,
                                document,
                                layer,
                                index,
                                selected,
                                thumbnails,
                                state,
                                &mut actions,
                            );
                            ui.end_row();
                        }
                    });
            });
        });
    actions
}

#[allow(clippy::too_many_arguments)]
fn layer_row(
    ui: &mut egui::Ui,
    document: &Document,
    layer: &Layer,
    index: usize,
    selected: usize,
    thumbnails: &HashMap<u64, egui::TextureHandle>,
    state: &mut LayersPanelState,
    actions: &mut Vec<LayerAction>,
) {
    let mut visible = layer.visible;
    if ui
        .checkbox(&mut visible, "")
        .on_hover_text("Show or hide this layer")
        .changed()
    {
        actions.push(LayerAction::SetVisible(index, visible));
    }

    let size = egui::Vec2::splat(THUMBNAIL_POINTS);
    match thumbnails.get(&layer.id) {
        Some(texture) => {
            ui.image((texture.id(), size));
        }
        None => {
            ui.allocate_space(size);
        }
    }

    if let Some((_, text)) = state.renaming.as_mut().filter(|(id, _)| *id == layer.id) {
        let response = ui.add(egui::TextEdit::singleline(text).desired_width(120.0));
        if !response.has_focus() && !response.lost_focus() {
            response.request_focus();
        }
        if response.lost_focus() {
            let cancelled = ui.input(|i| i.key_pressed(egui::Key::Escape));
            let name = text.trim().to_string();
            if !cancelled && !name.is_empty() && name != layer.name {
                actions.push(LayerAction::Rename(index, name));
            }
            state.renaming = None;
        }
    } else {
        let mut response = ui.selectable_label(index == selected, &layer.name);
        if !layer.is_base() {
            response = response.on_hover_text("Double-click to rename");
        }
        if response.clicked() {
            actions.push(LayerAction::Select(index));
        }
        if response.double_clicked() && !layer.is_base() {
            state.renaming = Some((layer.id, layer.name.clone()));
        }
    }

    // Keep the dropdown open while ticking several slots.
    let config = MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
    MenuButton::new(slot_summary(document, layer))
        .config(config)
        .ui(ui, |ui| {
            for (slot, material) in document.mesh.materials.iter().enumerate() {
                let mut on = layer.targets.get(slot).copied().unwrap_or(false);
                if ui.checkbox(&mut on, &material.name).changed() {
                    actions.push(LayerAction::SetTarget {
                        layer: index,
                        slot,
                        on,
                    });
                }
            }
        });
}

fn slot_summary(document: &Document, layer: &Layer) -> String {
    let ticked: Vec<usize> = (0..layer.targets.len())
        .filter(|&slot| layer.targets[slot])
        .collect();
    match ticked.as_slice() {
        [] => "No slots ⏷".to_string(),
        _ if ticked.len() == layer.targets.len() && ticked.len() > 1 => "All slots ⏷".to_string(),
        [slot] => format!("{} ⏷", document.mesh.materials[*slot].name),
        _ => format!("{} slots ⏷", ticked.len()),
    }
}

/// A small preview of the layer on its first ticked slot, over a checkerboard
/// so transparent areas are visible.
pub fn thumbnail(document: &Document, index: usize) -> egui::ColorImage {
    let layer = &document.layers[index];
    let image: Option<&Image> = (0..layer.targets.len())
        .filter(|&slot| layer.targets[slot])
        .find_map(|slot| {
            if layer.is_base() {
                document.mesh.materials.get(slot).map(|m| &m.base_color)
            } else {
                layer.textures.get(slot)?.as_ref()
            }
        });

    const SAMPLES: usize = 4;
    let n = THUMBNAIL_PIXELS;
    let mut pixels = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let checker = if (x / 6 + y / 6) % 2 == 0 {
                200.0
            } else {
                150.0
            };
            let mut sum = [0.0f32; 4];
            if let Some(image) = image {
                // Average a few samples across the area this pixel covers.
                for sy in 0..SAMPLES {
                    for sx in 0..SAMPLES {
                        let u = (x * SAMPLES + sx) as f32 / (n * SAMPLES) as f32;
                        let v = (y * SAMPLES + sy) as f32 / (n * SAMPLES) as f32;
                        let px = ((u * image.width as f32) as u32).min(image.width - 1);
                        let py = ((v * image.height as f32) as u32).min(image.height - 1);
                        let offset = ((py * image.width + px) * 4) as usize;
                        let sample = &image.rgba[offset..offset + 4];
                        // Weight colour by alpha so transparent pixels don't darken it.
                        let a = sample[3] as f32 / 255.0;
                        for c in 0..3 {
                            sum[c] += sample[c] as f32 * a;
                        }
                        sum[3] += sample[3] as f32;
                    }
                }
            }
            let count = (SAMPLES * SAMPLES) as f32;
            let alpha = sum[3] / count / 255.0;
            let blend = |c: usize| (sum[c] / count + checker * (1.0 - alpha)) as u8;
            pixels.push(egui::Color32::from_rgb(blend(0), blend(1), blend(2)));
        }
    }
    egui::ColorImage::new([n, n], pixels)
}
