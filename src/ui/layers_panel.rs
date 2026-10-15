use crate::document::{BlendMode, Document, FillMix, FillValues, Layer, LayerMask, MAX_BUMP_DEPTH};
use crate::generator::edges;
use crate::generator::{Axis, Generator, MAX_STEPS, Op, Pattern, Recipe};
use crate::mesh::{Channel, Image};
use crate::presets::Library;
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
        generator: Recipe,
    },
    /// Add the layers of preset `0` (counting the built-in ones first) above
    /// the selected layer.
    ApplyPreset(usize),
    /// Ask which layers to keep as a new preset.
    SavePreset,
    /// Delete the saved preset at this index (of `Library::saved`).
    DeletePreset(usize),
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
    /// The project's bump depth; `done` as for `Opacity`.
    BumpDepth {
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
    draft: Option<(u64, Recipe)>,
    /// The same for the pattern of a fill layer's second colour.
    mix_draft: Option<(u64, Recipe)>,
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
    presets: &Library,
) -> Vec<LayerAction> {
    let mut actions = Vec::new();
    let count = document.layers.len();
    let current = &document.layers[selected];

    // The selected layer's settings live in their own panel below the list,
    // so a long stack of generators can't push the layers around.
    egui::Panel::bottom("details_panel")
        .resizable(true)
        .default_size(ui.available_height() * 0.55)
        .show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("Details");
                ui.weak(&current.name);
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    super::fit_widths(ui);
                    details(ui, current, selected, state, &mut actions);
                    if current.is_base() {
                        model_settings(ui, document.bump_depth, &mut actions);
                    }
                });
        });

    ui.heading("Layers");
    ui.separator();
    layer_buttons(ui, current, (selected, count), presets, &mut actions);
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            // Top layer first, like other paint programs.
            for index in (0..count).rev() {
                let layer = &document.layers[index];
                ui.horizontal(|ui| {
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
                });
            }
        });
    actions
}

/// The selected layer's blend mode and opacity, then what makes its pixels.
fn details(
    ui: &mut egui::Ui,
    current: &Layer,
    selected: usize,
    state: &mut LayersPanelState,
    actions: &mut Vec<LayerAction>,
) {
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
    settings(ui, current, selected, state, actions);
}

/// Settings of the whole model, shown with the base layer.
fn model_settings(ui: &mut egui::Ui, bump_depth: f32, actions: &mut Vec<LayerAction>) {
    ui.separator();
    ui.strong("Model");
    let mut value = bump_depth;
    let slider = ui
        .add(
            egui::Slider::new(&mut value, 0.0..=MAX_BUMP_DEPTH)
                .custom_formatter(|v, _| format!("{:.1}%", v * 100.0))
                .custom_parser(|text| {
                    let number = text
                        .trim()
                        .trim_end_matches('%')
                        .trim()
                        .parse::<f64>()
                        .ok()?;
                    Some(number / 100.0)
                })
                .text("Bump depth"),
        )
        .on_hover_text(
            "How far a height of 1 stands out from a height of 0, as a share of \
             the model's size. Applies to the viewport and to exported normal maps.",
        );
    if slider.changed() {
        actions.push(LayerAction::BumpDepth {
            value,
            done: !slider.dragged(),
        });
    }
    if slider.drag_stopped() {
        actions.push(LayerAction::BumpDepth { value, done: true });
    }
}

/// Height of the buttons above the layer list, like the brush tool buttons.
const BUTTON_HEIGHT: f32 = 28.0;

/// Add (layers, presets and masks), delete and move, for the selected layer.
fn layer_buttons(
    ui: &mut egui::Ui,
    current: &Layer,
    (selected, count): (usize, usize),
    presets: &Library,
    actions: &mut Vec<LayerAction>,
) {
    let editable = !current.is_base();
    let spacing = ui.spacing().item_spacing.x;
    let room = ui.available_width() - 3.0 * spacing;
    let (wide, narrow) = (room * 0.32, room * 0.18);
    let size = |width: f32| egui::vec2(width, BUTTON_HEIGHT);
    ui.horizontal(|ui| {
        let add = egui::Button::new("Add ⏷").min_size(size(wide));
        let (response, _) = MenuButton::from_button(add)
            .ui(ui, |ui| add_menu(ui, current, selected, presets, actions));
        response.on_hover_text(
            "Add layers and presets above the selected layer, or work with its mask",
        );

        let delete = ui.add_enabled(editable, egui::Button::new("Delete").min_size(size(wide)));
        if delete.on_hover_text("Delete the selected layer").clicked() {
            actions.push(LayerAction::Delete(selected));
        }
        let up = egui::Button::new("⬆").min_size(size(narrow));
        let up = ui.add_enabled(editable && selected + 1 < count, up);
        if up.on_hover_text("Move up").clicked() {
            actions.push(LayerAction::Move {
                from: selected,
                to: selected + 1,
            });
        }
        let down = egui::Button::new("⬇").min_size(size(narrow));
        let down = ui.add_enabled(editable && selected > 1, down);
        if down.on_hover_text("Move down").clicked() {
            actions.push(LayerAction::Move {
                from: selected,
                to: selected - 1,
            });
        }
    });
}

/// What the Add button's menu offers: new layers, then the selected layer's mask.
fn add_menu(
    ui: &mut egui::Ui,
    current: &Layer,
    selected: usize,
    presets: &Library,
    actions: &mut Vec<LayerAction>,
) {
    ui.weak("Layers");
    if ui
        .button("Paint layer")
        .on_hover_text("An empty layer to paint on, above the selected one")
        .clicked()
    {
        actions.push(LayerAction::Add);
    }
    if ui
        .button("Fill layer")
        .on_hover_text("The brush's values everywhere, shown through a mask")
        .clicked()
    {
        actions.push(LayerAction::AddFill);
    }
    presets_menu(ui, presets, actions);

    ui.separator();
    ui.weak("Mask of the selected layer");
    let has_mask = current.mask.is_some();
    ui.add_enabled_ui(!current.is_base(), |ui| {
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
            "A mask made from noise or a gradient, set in the layer's details",
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
    .on_disabled_hover_text("The base layer can't have a mask");
}

/// Adds, saves and deletes presets.
fn presets_menu(ui: &mut egui::Ui, presets: &Library, actions: &mut Vec<LayerAction>) {
    ui.menu_button("Presets", |ui| {
        let built_in = presets.built_in().len();
        let all = presets.built_in().iter().chain(presets.saved());
        for (index, preset) in all.enumerate() {
            if index == built_in {
                ui.separator();
            }
            let layers = preset.layers.len();
            let hint = format!(
                "Add {layers} fill layer{} above the selected one",
                if layers == 1 { "" } else { "s" }
            );
            if ui.button(&preset.name).on_hover_text(hint).clicked() {
                actions.push(LayerAction::ApplyPreset(index));
            }
        }
        ui.separator();
        if ui
            .button("Save layers as preset...")
            .on_hover_text("Keep fill layers' settings to add to any model later")
            .clicked()
        {
            actions.push(LayerAction::SavePreset);
        }
        ui.add_enabled_ui(!presets.saved().is_empty(), |ui| {
            ui.menu_button("Delete preset", |ui| {
                for (index, preset) in presets.saved().iter().enumerate() {
                    if ui.button(&preset.name).clicked() {
                        actions.push(LayerAction::DeletePreset(index));
                    }
                }
            });
        });
    });
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
                    Channel::Height => {
                        ui.add(egui::Slider::new(&mut edited.height, 0.0..=1.0));
                    }
                }
            });
        }
        let mut mixed = edited.mix.is_some();
        if ui
            .checkbox(&mut mixed, "Mix in a second look")
            .on_hover_text("Blend to other values where a pattern is white")
            .changed()
        {
            edited.mix = mixed.then(|| FillMix {
                color: edited.color.map(|c| c / 2),
                roughness: edited.roughness,
                metallic: edited.metallic,
                height: edited.height,
                ..FillMix::default()
            });
            state.mix_draft = None;
        }
        // The pattern takes a moment to make, so its edits wait for a let-go;
        // the colours blend live.
        let mut released = false;
        let applied = fill.mix.map(|mix| mix.generator);
        if let Some(mix) = edited.mix.as_mut() {
            ui.horizontal(|ui| {
                ui.label("Colour");
                ui.color_edit_button_srgb(&mut mix.color);
            });
            ui.add(egui::Slider::new(&mut mix.roughness, 0.0..=1.0).text("Roughness"));
            ui.add(egui::Slider::new(&mut mix.metallic, 0.0..=1.0).text("Metallic"));
            ui.add(egui::Slider::new(&mut mix.height, 0.0..=1.0).text("Height"));
            ui.label("Where the pattern is white:");
            if let Some((id, draft)) = state.mix_draft
                && id == layer.id
            {
                mix.generator = draft;
            }
            recipe_controls(ui, "mix", &mut mix.generator, &mut released);
        }
        let wanted = edited.mix.map(|mix| mix.generator);
        if released || mixed != fill.mix.is_some() {
            state.mix_draft = None;
        } else if wanted != applied
            && let Some(wanted) = wanted
        {
            state.mix_draft = Some((layer.id, wanted));
            // Everything but the pattern still applies at once.
            if let (Some(mix), Some(applied)) = (edited.mix.as_mut(), applied) {
                mix.generator = applied;
            }
        }
        if edited != fill {
            actions.push(LayerAction::Fill {
                layer: index,
                fill: edited,
            });
        }
        if state.mix_draft.is_some() {
            ui.weak("Applies when you let go.");
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
        recipe_controls(ui, "mask", &mut edited, &mut released);
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

/// The controls of a recipe: its first pattern, the patterns combined into
/// it, and a button to combine one more. Sets `released` like
/// `generator_controls`, and at once when patterns are added, removed or
/// combined differently.
fn recipe_controls(ui: &mut egui::Ui, id: &str, recipe: &mut Recipe, released: &mut bool) {
    ui.push_id(id, |ui| {
        let combined = recipe.step_count() > 0;
        if combined {
            ui.strong("Pattern 1");
        }
        generator_controls(ui, "first", &mut recipe.base, released);
        let mut remove = None;
        for index in 0..recipe.step_count() {
            let Some(step) = recipe.step_mut(index) else {
                break;
            };
            ui.push_id(index, |ui| {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.strong(format!("Pattern {}", index + 2));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Remove").clicked() {
                            remove = Some(index);
                        }
                    });
                });
                let before = step.op;
                egui::ComboBox::from_label("Combine")
                    .selected_text(step.op.name())
                    .show_ui(ui, |ui| {
                        for op in Op::ALL {
                            ui.selectable_value(&mut step.op, op, op.name());
                        }
                    })
                    .response
                    .on_hover_text(
                        "Multiply cuts holes in what came before, Add and Lighter grow it, \
                         Subtract and Darker carve it away",
                    );
                *released |= step.op != before;
                let amount = ui.add(egui::Slider::new(&mut step.amount, 0.0..=1.0).text("Amount"));
                *released |= amount.drag_stopped() || (amount.changed() && !amount.dragged());
                generator_controls(ui, "pattern", &mut step.generator, released);
            });
        }
        if let Some(index) = remove {
            recipe.remove_step(index);
            *released = true;
        }
        if recipe.step_count() < MAX_STEPS
            && ui
                .button("+ Combine another pattern")
                .on_hover_text("Mix a second pattern into this one, e.g. fine grain into blotches")
                .clicked()
        {
            recipe.add_step();
            *released = true;
        }
    });
}

/// The pattern's controls. Sets `released` once the user has let go of one
/// (or picked something that isn't dragged), which is when to apply them.
fn generator_controls(ui: &mut egui::Ui, id: &str, edited: &mut Generator, released: &mut bool) {
    // A layer can show two sets of these, whose combo boxes share labels.
    ui.push_id(id, |ui| generator_widgets(ui, edited, released));
}

fn generator_widgets(ui: &mut egui::Ui, edited: &mut Generator, released: &mut bool) {
    let shown = *edited;
    let let_go = |response: egui::Response| {
        response.drag_stopped() || (response.changed() && !response.dragged())
    };
    egui::ComboBox::from_label("Pattern")
        .selected_text(edited.pattern.name())
        .show_ui(ui, |ui| {
            for pattern in Pattern::ALL {
                ui.selectable_value(&mut edited.pattern, pattern, pattern.name());
            }
        });
    if edited.pattern.has_axis() {
        let label = match edited.pattern {
            Pattern::Stripes => "Across",
            Pattern::Bricks | Pattern::Tiles => "Rows along",
            _ => "Axis",
        };
        egui::ComboBox::from_label(label)
            .selected_text(edited.axis.name())
            .show_ui(ui, |ui| {
                for axis in Axis::ALL {
                    ui.selectable_value(&mut edited.axis, axis, axis.name());
                }
            });
    }
    if edited.pattern.fold().is_some() {
        let label = match edited.pattern {
            Pattern::Cavity => "Crease width",
            _ => "Edge width",
        };
        let response = ui.add(
            egui::Slider::new(&mut edited.edge_width, 0.002..=edges::MAX_REACH)
                .logarithmic(true)
                .text(label),
        );
        *released |= let_go(response);
        let response = ui.add(egui::Slider::new(&mut edited.breakup, 0.0..=1.0).text("Breakup"));
        *released |= let_go(response);
    }
    if edited.pattern != Pattern::Gradient {
        let response = ui.add(
            egui::Slider::new(&mut edited.scale, 0.25..=64.0)
                .logarithmic(true)
                .text("Scale"),
        );
        *released |= let_go(response);
    }
    if edited.pattern.is_tiling() {
        if edited.pattern != Pattern::Checker {
            let label = match edited.pattern {
                Pattern::Stripes => "Line width",
                _ => "Mortar width",
            };
            let response = ui.add(egui::Slider::new(&mut edited.line_width, 0.0..=1.0).text(label));
            *released |= let_go(response);
        }
        let response = ui
            .add(egui::Slider::new(&mut edited.bevel, 0.0..=0.5).text("Bevel"))
            .on_hover_text("Soften the edges; as a height, a rounded bevel");
        *released |= let_go(response);
    }
    if edited.pattern.is_noisy() {
        let response = ui.add(egui::Slider::new(&mut edited.octaves, 1..=8).text("Detail"));
        *released |= let_go(response);
        let response = ui.add(egui::DragValue::new(&mut edited.seed).prefix("Seed "));
        *released |= let_go(response);
    }
    let response = ui.add(egui::Slider::new(&mut edited.low, 0.0..=1.0).text("Black at"));
    *released |= let_go(response);
    let response = ui.add(egui::Slider::new(&mut edited.high, 0.0..=1.0).text("White at"));
    *released |= let_go(response);
    if ui.checkbox(&mut edited.invert, "Invert").changed() {
        *released = true;
    }
    if edited.pattern != shown.pattern || edited.axis != shown.axis {
        *released = true;
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
    // The slot dropdown sits at the right edge, whatever the panel's width.
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
