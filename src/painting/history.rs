//! Undo / redo.
//!
//! Each step is a list of `Change`s (a stroke can touch several slots; most
//! layer edits are one change). Applying a change turns it into its inverse,
//! so undo applies a step's changes in reverse order and keeps them for redo,
//! and redo applies them forward again.

use crate::document::{Change, Dirty, Document};

/// Edge length, in texels, of the tiles a stroke's undo step stores.
pub const TILE_SIZE: u32 = 64;
/// Memory the history may use before the oldest steps are dropped.
const DEFAULT_BUDGET_BYTES: usize = 512 * 1024 * 1024;

struct Edit {
    id: u64,
    /// Shown in the history panel, e.g. "Paint on Layer 2".
    label: String,
    changes: Vec<Change>,
}

impl Edit {
    fn bytes(&self) -> usize {
        self.changes.iter().map(Change::bytes).sum()
    }
}

pub struct History {
    undo: Vec<Edit>,
    redo: Vec<Edit>,
    next_id: u64,
    bytes: usize,
    budget_bytes: usize,
    /// Oldest steps dropped to stay within the budget.
    dropped: usize,
}

impl Default for History {
    fn default() -> Self {
        Self::with_budget(DEFAULT_BUDGET_BYTES)
    }
}

impl History {
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            next_id: 1,
            bytes: 0,
            budget_bytes,
            dropped: 0,
        }
    }

    /// Records changes that were just made, as one step called `label`.
    /// Clears the redo steps.
    pub fn push(&mut self, label: impl Into<String>, changes: Vec<Change>) {
        if changes.is_empty() {
            return;
        }
        let edit = Edit {
            id: self.next_id,
            label: label.into(),
            changes,
        };
        self.next_id += 1;
        self.bytes -= self.redo.drain(..).map(|edit| edit.bytes()).sum::<usize>();
        self.bytes += edit.bytes();
        self.undo.push(edit);
        // Drop the oldest steps once over budget, but always keep the newest.
        while self.bytes > self.budget_bytes && self.undo.len() > 1 {
            self.bytes -= self.undo.remove(0).bytes();
            self.dropped += 1;
        }
    }

    /// Reverts the latest step. Returns what needs recompositing.
    pub fn undo(&mut self, document: &mut Document) -> Vec<Dirty> {
        let Some(mut edit) = self.undo.pop() else {
            return Vec::new();
        };
        let dirty = edit
            .changes
            .iter_mut()
            .rev()
            .filter_map(|change| document.apply(change))
            .collect();
        self.redo.push(edit);
        dirty
    }

    /// Re-applies the latest undone step.
    pub fn redo(&mut self, document: &mut Document) -> Vec<Dirty> {
        let Some(mut edit) = self.redo.pop() else {
            return Vec::new();
        };
        let dirty = edit
            .changes
            .iter_mut()
            .filter_map(|change| document.apply(change))
            .collect();
        self.undo.push(edit);
        dirty
    }

    /// Undoes or redoes steps until `position` steps are applied (see
    /// `labels`). Returns what needs recompositing.
    pub fn go_to(&mut self, document: &mut Document, position: usize) -> Vec<Dirty> {
        let mut dirty = Vec::new();
        while self.undo.len() > position && self.can_undo() {
            dirty.extend(self.undo(document));
        }
        while self.undo.len() < position && self.can_redo() {
            dirty.extend(self.redo(document));
        }
        dirty
    }

    /// Every step's label, oldest first: the applied steps, then the undone
    /// ones in the order redo would bring them back.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.undo
            .iter()
            .chain(self.redo.iter().rev())
            .map(|edit| edit.label.as_str())
    }

    /// How many of `labels` are applied.
    pub fn position(&self) -> usize {
        self.undo.len()
    }

    /// How many of the oldest steps were dropped to save memory.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// Memory used by the stored steps, and the most it may use.
    pub fn memory(&self) -> (usize, usize) {
        (self.bytes, self.budget_bytes)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Identifies the current state, to tell whether it matches the last save.
    pub fn state(&self) -> u64 {
        self.undo.last().map_or(0, |edit| edit.id)
    }

    pub fn clear(&mut self) {
        *self = Self::with_budget(self.budget_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::History;
    use crate::document::{CLEAR, Change, Document, Tile};
    use crate::mesh::Channel;
    use crate::mesh::{Image, ImageRegion, primitives};

    fn document() -> Document {
        Document::new(primitives::cube())
    }

    /// Paints `region` of layer 1 with `value`; returns the undoing change.
    fn paint(document: &mut Document, region: ImageRegion, value: u8) -> Change {
        let (w, h) = document.slot_sizes()[0];
        let layer = &mut document.layers[1];
        let image = layer.textures[0][0].get_or_insert_with(|| Image::filled(w, h, CLEAR));
        let mut before = Vec::new();
        for y in region.y..region.y + region.height {
            for x in region.x..region.x + region.width {
                let offset = ((y * image.width + x) * 4) as usize;
                before.extend_from_slice(&image.rgba[offset..offset + 4]);
                image.rgba[offset..offset + 4].fill(value);
            }
        }
        let id = layer.id;
        Change::Pixels {
            layer: id,
            material: 0,
            channel: Channel::BaseColor,
            tiles: vec![Tile {
                region,
                pixels: before,
            }],
        }
    }

    fn composite(document: &Document) -> Vec<u8> {
        document
            .composite(0, Channel::BaseColor)
            .unwrap()
            .rgba
            .clone()
    }

    #[test]
    fn undo_and_redo_restore_each_state() {
        let mut document = document();
        let mut history = History::default();
        let blank = composite(&document);

        let first = paint(
            &mut document,
            ImageRegion {
                x: 2,
                y: 1,
                width: 5,
                height: 3,
            },
            100,
        );
        history.push("Test", vec![first]);
        let after_first = composite(&document);
        history.push("Test", vec![document.insert_new_layer(2)]);
        let second = paint(
            &mut document,
            ImageRegion {
                x: 4,
                y: 2,
                width: 9,
                height: 6,
            },
            200,
        );
        history.push("Test", vec![second]);
        let after_second = composite(&document);

        assert!(!history.undo(&mut document).is_empty());
        history.undo(&mut document);
        assert_eq!(document.layers.len(), 2);
        assert_eq!(composite(&document), after_first);
        history.undo(&mut document);
        assert_eq!(composite(&document), blank);
        assert!(history.undo(&mut document).is_empty());

        history.redo(&mut document);
        assert_eq!(composite(&document), after_first);
        history.redo(&mut document);
        history.redo(&mut document);
        assert_eq!(composite(&document), after_second);
        assert!(!history.can_redo());
    }

    #[test]
    fn a_new_edit_clears_redo() {
        let mut document = document();
        let mut history = History::default();
        let change = paint(&mut document, ImageRegion::pixel(1, 1), 50);
        history.push("Test", vec![change]);
        history.undo(&mut document);
        assert!(history.can_redo());

        let change = paint(&mut document, ImageRegion::pixel(2, 2), 60);
        history.push("Test", vec![change]);
        assert!(!history.can_redo());
    }

    #[test]
    fn oldest_steps_are_dropped_over_budget() {
        let mut document = document();
        // Each step stores 4 x 4 pixels = 64 bytes; room for two.
        let mut history = History::with_budget(128);
        let region = ImageRegion {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        };
        for value in 1..=3 {
            let change = paint(&mut document, region, value);
            history.push("Test", vec![change]);
        }
        assert!(!history.undo(&mut document).is_empty());
        assert!(!history.undo(&mut document).is_empty());
        assert!(history.undo(&mut document).is_empty());
        // Back to the state after the first (forgotten) step.
        assert_eq!(
            document.layers[1].textures[0][0].as_ref().unwrap().rgba[0],
            1
        );
    }

    #[test]
    fn state_tracks_the_saved_point() {
        let mut document = document();
        let mut history = History::default();
        let change = paint(&mut document, ImageRegion::pixel(0, 0), 1);
        history.push("Test", vec![change]);
        let saved = history.state();

        let change = paint(&mut document, ImageRegion::pixel(1, 0), 2);
        history.push("Test", vec![change]);
        assert_ne!(history.state(), saved);
        history.undo(&mut document);
        assert_eq!(history.state(), saved);
    }

    #[test]
    fn go_to_jumps_several_steps_and_keeps_the_labels() {
        let mut document = document();
        let mut history = History::default();
        let mut states = vec![composite(&document)];
        for (step, value) in [(0, 50), (1, 100), (2, 150)] {
            let region = ImageRegion {
                x: step * 8,
                y: 0,
                width: 4,
                height: 4,
            };
            let change = paint(&mut document, region, value);
            history.push(format!("Step {step}"), vec![change]);
            states.push(composite(&document));
        }
        let labels = |history: &History| history.labels().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(labels(&history), ["Step 0", "Step 1", "Step 2"]);

        history.go_to(&mut document, 1);
        assert_eq!(history.position(), 1);
        assert!(composite(&document) == states[1]);
        // Undone steps stay listed, in order, until something new is done.
        assert_eq!(labels(&history), ["Step 0", "Step 1", "Step 2"]);

        history.go_to(&mut document, 3);
        assert!(composite(&document) == states[3]);
        history.go_to(&mut document, 0);
        assert!(composite(&document) == states[0]);
        // Out-of-range positions stop at the ends.
        history.go_to(&mut document, 99);
        assert_eq!(history.position(), 3);
    }
}
