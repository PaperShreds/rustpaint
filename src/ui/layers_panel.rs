use crate::document::{BlendMode, Document, FillValues, Layer, LayerMask};
use crate::generator::edges;
use crate::generator::{Axis, Generator, Pattern};
use crate::mesh::{Channel, Image};
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
    /// A fill layer, with the brush's current values.
    AddFill,
    /// Give the layer a mask made by a generator.
    AddGeneratorMask(usize),
    /// Keep the generated mask's pixels, so it can be painted on.
    ConvertGeneratorMask(usize),
    /// New fill values (the layer's images follow them live).
    Fill {
        layer: usize,
        fill: FillValues,
    },
    /// New generator settings, once the user has let go of the control.
    Generator {
        layer: usize,
        generator: Generator,
    },
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
    SetBlend(usize, BlendMode),
    /// Select the layer, and paint on its mask (true) or its channels.
    EditMask(usize, bool),
    /// Give the layer a mask that shows it all, or (`hide`) hides it all.
    AddMask {
        layer: usize,
        hide: bool,
    },
    DeleteMask(usize),
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
    /// Generator settings being dragged: they apply once the control is
    /// let go, because making the pattern takes a moment.
    draft: Option<(u64, Generator)>,
}

/// Layer thumbnails by (layer id, whether it shows the mask).
pub type Thumbnails = HashMap<(u64, bool), egui::TextureHandle>;

/// Draws the layers panel's contents into `ui`. `on_mask` says whether the
/// selected layer's mask, rather than its paint, is being painted.
pub fn show(
    ui: &mut egui::Ui,
    document: &Document,
    selected: usize,
    on_mask: bool,
    thumbnails: &Thumbnails,
    state: &mut LayersPanelState,
) -> Vec<LayerAction> {
    let mut actions = Vec::new();
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
        if ui
            .button("+ Fill")
            .on_hover_text("Add a fill layer: the brush's values everywhere, shown through a mask")
            .clicked()
        {
            actions.push(LayerAction::AddFill);
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
        let has_mask = current.mask.is_some();
        ui.add_enabled_ui(editable, |ui| {
            ui.menu_button("Mask ⏷", |ui| {
                let add = |ui: &mut egui::Ui, label, hint| {
                    ui.add_enabled(!has_mask, egui::Button::new(label))
                        .on_hover_text(hint)
                        .clicked()
                };
                if add(
                    ui,
                    "Add mask (show all)",
                    "Then paint with Erase on to hide parts",
                ) {
                    actions.push(LayerAction::AddMask {
                        layer: selected,
                        hide: false,
                    });
                }
                if add(ui, "Add mask (hide all)", "Then paint to reveal parts") {
                    actions.push(LayerAction::AddMask {
                        layer: selected,
                        hide: true,
                    });
                }
                if add(
                    ui,
                    "Add generator mask",
                    "A mask made from noise or a gradient, set in the layer's settings",
                ) {
                    actions.push(LayerAction::AddGeneratorMask(selected));
                }
                let generated = current.mask_generator.is_some();
                if ui
                    .add_enabled(generated, egui::Button::new("Convert to painted mask"))
                    .on_hover_text("Keep the pattern's pixels and paint on them")
                    .clicked()
                {
                    actions.push(LayerAction::ConvertGeneratorMask(selected));
                }
                ui.separator();
                if ui
                    .add_enabled(has_mask, egui::Button::new("Delete mask"))
                    .clicked()
                {
                    actions.push(LayerAction::DeleteMask(selected));
                }
            })
            .response
            .on_hover_text("A mask hides parts of a layer without erasing its paint")
            .on_disabled_hover_text("The base layer can't have a mask");
        });
    });

    let mut blend = current.blend;
    egui::ComboBox::from_label("Blend")
        .selected_text(blend.name())
        .show_ui(ui, |ui| {
            for mode in BlendMode::ALL {
                ui.selectable_value(&mut blend, mode, mode.name());
            }
        });
    if blend != current.blend {
        actions.push(LayerAction::SetBlend(selected, blend));
    }

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
    settings(ui, current, selected, state, &mut actions);
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
                        (selected, on_mask),
                        thumbnails,
                        state,
                        &mut actions,
                    );
                    ui.end_row();
                }
            });
    });
    actions
}

/// Controls for what makes the layer's pixels: a fill layer's values and a
/// generated mask's pattern.
fn settings(
    ui: &mut egui::Ui,
    layer: &Layer,
    index: usize,
    state: &mut LayersPanelState,
    actions: &mut Vec<LayerAction>,
) {
    if let Some(fill) = layer.fill {
        ui.separator();
        ui.strong("Fill");
        let mut edited = fill;
        for channel in Channel::ALL {
            ui.horizontal(|ui| {
                ui.checkbox(&mut edited.channels[channel.index()], channel.name());
                match channel {
                    Channel::BaseColor => {
                        ui.color_edit_button_srgb(&mut edited.color);
                    }
                    Channel::Roughness => {
                        ui.add(egui::Slider::new(&mut edited.roughness, 0.0..=1.0));
                    }
                    Channel::Metallic => {
                        ui.add(egui::Slider::new(&mut edited.metallic, 0.0..=1.0));
                    }
                }
            });
        }
        if edited != fill {
            actions.push(LayerAction::Fill {
                layer: index,
                fill: edited,
            });
        }
        if layer.mask.is_none() {
            ui.weak("Add a mask to paint where this layer shows.");
        }
    }
    if let Some(applied) = layer.mask_generator {
        ui.separator();
        ui.strong("Mask generator");
        let shown = match state.draft {
            Some((id, draft)) if id == layer.id => draft,
            _ => applied,
        };
        let mut edited = shown;
        // Whether the user let go of a control, and whether something changed.
        let mut released = false;
        let mut watch = |response: egui::Response| {
            released |= response.drag_stopped() || (response.changed() && !response.dragged());
        };
        egui::ComboBox::from_label("Pattern")
            .selected_text(edited.pattern.name())
            .show_ui(ui, |ui| {
                for pattern in Pattern::ALL {
                    ui.selectable_value(&mut edited.pattern, pattern, pattern.name());
                }
            });
        if edited.pattern == Pattern::Gradient {
            egui::ComboBox::from_label("Axis")
                .selected_text(edited.axis.name())
                .show_ui(ui, |ui| {
                    for axis in Axis::ALL {
                        ui.selectable_value(&mut edited.axis, axis, axis.name());
                    }
                });
        } else {
            if edited.pattern == Pattern::EdgeWear {
                let response = ui.add(
                    egui::Slider::new(&mut edited.edge_width, 0.002..=edges::MAX_REACH)
                        .logarithmic(true)
                        .text("Edge width"),
                );
                watch(response);
                let response =
                    ui.add(egui::Slider::new(&mut edited.breakup, 0.0..=1.0).text("Breakup"));
                watch(response);
            }
            let response = ui.add(
                egui::Slider::new(&mut edited.scale, 0.25..=64.0)
                    .logarithmic(true)
                    .text("Scale"),
            );
            watch(response);
            let response = ui.add(egui::Slider::new(&mut edited.octaves, 1..=8).text("Detail"));
            watch(response);
            let response = ui.add(egui::DragValue::new(&mut edited.seed).prefix("Seed "));
            watch(response);
        }
        let response = ui.add(egui::Slider::new(&mut edited.low, 0.0..=1.0).text("Black at"));
        watch(response);
        let response = ui.add(egui::Slider::new(&mut edited.high, 0.0..=1.0).text("White at"));
        watch(response);
        if ui.checkbox(&mut edited.invert, "Invert").changed() {
            released = true;
        }
        if edited.pattern != shown.pattern || edited.axis != shown.axis {
            released = true;
        }
        if released {
            state.draft = None;
            if edited != applied {
                actions.push(LayerAction::Generator {
                    layer: index,
                    generator: edited,
                });
            }
        } else if edited != shown {
            state.draft = Some((layer.id, edited));
        }
        if state.draft.is_some() {
            ui.weak("Applies when you let go.");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn layer_row(
    ui: &mut egui::Ui,
    document: &Document,
    layer: &Layer,
    index: usize,
    (selected, on_mask): (usize, bool),
    thumbnails: &Thumbnails,
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

    // The layer's paint and, if it has one, its mask. Clicking one picks
    // what the brush paints on; that one is framed.
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let parts = [
            (false, "Paint on the layer"),
            (
                true,
                "Paint on the mask: white shows the layer, black hides it",
            ),
        ];
        for (mask, hint) in parts {
            if mask && layer.mask.is_none() {
                continue;
            }
            let size = egui::Vec2::splat(THUMBNAIL_POINTS);
            let response = match thumbnails.get(&(layer.id, mask)) {
                Some(texture) => {
                    ui.add(egui::Image::new((texture.id(), size)).sense(egui::Sense::click()))
                }
                None => ui.allocate_response(size, egui::Sense::click()),
            };
            let active = index == selected && on_mask == mask && layer.mask.is_some();
            if active {
                let stroke = ui.visuals().selection.stroke;
                ui.painter().rect_stroke(
                    response.rect.expand(2.0),
                    2.0,
                    egui::Stroke::new(2.0, stroke.color),
                    egui::StrokeKind::Outside,
                );
            }
            if !layer.is_base() && response.on_hover_text(hint).clicked() {
                actions.push(LayerAction::EditMask(index, mask));
            }
        }
    });

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

/// A greyscale preview of the layer's mask on its first ticked slot, or
/// `None` without a mask.
pub fn mask_thumbnail(document: &Document, index: usize) -> Option<egui::ColorImage> {
    let layer = &document.layers[index];
    layer.mask?;
    let slot = (0..layer.targets.len())
        .find(|&slot| layer.targets[slot])
        .unwrap_or(0);
    let n = THUMBNAIL_PIXELS;
    let value = |x: usize, y: usize| match document.layer_mask(layer, slot) {
        LayerMask::None => 255,
        LayerMask::Uniform(value) => value,
        LayerMask::Image(image) => {
            let px = ((x as u32 * image.width) / n as u32).min(image.width - 1);
            let py = ((y as u32 * image.height) / n as u32).min(image.height - 1);
            image.rgba[((py * image.width + px) * 4) as usize]
        }
    };
    let pixels = (0..n * n)
        .map(|i| egui::Color32::from_gray(value(i % n, i / n)))
        .collect();
    Some(egui::ColorImage::new([n, n], pixels))
}

/// A small preview of the layer on its first ticked slot (base colour, or
/// another channel if that's all it has), over a checkerboard so transparent
/// areas are visible.
pub fn thumbnail(document: &Document, index: usize) -> egui::ColorImage {
    let layer = &document.layers[index];
    let image: Option<&Image> = (0..layer.targets.len())
        .filter(|&slot| layer.targets[slot])
        .find_map(|slot| {
            Channel::ALL
                .into_iter()
                .find_map(|channel| document.layer_image(layer, slot, channel))
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
