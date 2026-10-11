//! Preferences remembered between runs: recent files, the environment, the
//! view and the brush.
//!
//! Stored through eframe's storage (a small file in the user's data folder)
//! as plain `key = value` lines. Unknown keys and unreadable values are
//! skipped, so an old or hand-edited file never stops the program starting.

use crate::painting::brush::Brush;
use crate::renderer::environment::{BuiltIn, EnvironmentSource};
use crate::renderer::viewport::{EnvironmentSettings, ViewMode};
use std::path::{Path, PathBuf};

/// The key under which eframe stores the settings.
pub const STORAGE_KEY: &str = "rustpaint_settings";
/// How many files File > Open recent lists.
pub const MAX_RECENT_FILES: usize = 10;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings {
    /// Most recent first.
    pub recent_files: Vec<PathBuf>,
    pub environment_source: EnvironmentSource,
    pub environment: EnvironmentSettings,
    pub view_mode: ViewMode,
    pub brush: Brush,
}

impl Settings {
    pub fn load(storage: Option<&dyn eframe::Storage>) -> Self {
        storage
            .and_then(|storage| storage.get_string(STORAGE_KEY))
            .map_or_else(Self::default, |text| Self::from_text(&text))
    }

    pub fn save(&self, storage: &mut dyn eframe::Storage) {
        storage.set_string(STORAGE_KEY, self.to_text());
    }

    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        let mut put = |key: &str, value: String| lines.push(format!("{key} = {value}"));
        for path in &self.recent_files {
            put("recent", path.display().to_string());
        }
        put(
            "environment",
            match &self.environment_source {
                EnvironmentSource::BuiltIn(built_in) => built_in.key().to_string(),
                EnvironmentSource::File(path) => format!("file:{}", path.display()),
            },
        );
        let environment = &self.environment;
        put(
            "environment_rotation",
            environment.rotation_degrees.to_string(),
        );
        put("environment_intensity", environment.intensity.to_string());
        put(
            "environment_background",
            environment.show_background.to_string(),
        );
        put(
            "environment_background_blur",
            environment.background_blur.to_string(),
        );
        let view = ViewMode::ALL
            .iter()
            .position(|mode| *mode == self.view_mode);
        put("view_mode", view.unwrap_or(0).to_string());
        let brush = &self.brush;
        put("brush_size", brush.size.to_string());
        put("brush_hardness", brush.hardness.to_string());
        put("brush_opacity", brush.opacity.to_string());
        put("brush_paint_color", brush.paint_color.to_string());
        put("brush_color", brush.color.map(|c| c.to_string()).join(" "));
        put("brush_paint_roughness", brush.paint_roughness.to_string());
        put("brush_roughness", brush.roughness.to_string());
        put("brush_paint_metallic", brush.paint_metallic.to_string());
        put("brush_metallic", brush.metallic.to_string());
        put("brush_screen_space", brush.screen_space.to_string());
        put("brush_paint_over_edges", brush.paint_over_edges.to_string());
        lines.join("\n") + "\n"
    }

    pub fn from_text(text: &str) -> Self {
        let mut settings = Self::default();
        for line in text.lines() {
            let Some((key, value)) = line.split_once(" = ") else {
                continue;
            };
            settings.set(key.trim(), value);
        }
        settings
    }

    /// Applies one line; a value that doesn't parse leaves the default.
    fn set(&mut self, key: &str, value: &str) {
        fn number(value: &str, target: &mut f32, range: std::ops::RangeInclusive<f32>) {
            if let Ok(parsed) = value.trim().parse::<f32>()
                && parsed.is_finite()
            {
                *target = parsed.clamp(*range.start(), *range.end());
            }
        }
        fn flag(value: &str, target: &mut bool) {
            if let Ok(parsed) = value.trim().parse() {
                *target = parsed;
            }
        }
        let environment = &mut self.environment;
        let brush = &mut self.brush;
        match key {
            "recent" if self.recent_files.len() < MAX_RECENT_FILES && !value.is_empty() => {
                self.recent_files.push(PathBuf::from(value));
            }
            "environment" => {
                let source = match value.strip_prefix("file:") {
                    Some(path) => Some(EnvironmentSource::File(PathBuf::from(path))),
                    None => BuiltIn::from_key(value.trim()).map(EnvironmentSource::BuiltIn),
                };
                if let Some(source) = source {
                    self.environment_source = source;
                }
            }
            "environment_rotation" => number(value, &mut environment.rotation_degrees, 0.0..=360.0),
            "environment_intensity" => number(value, &mut environment.intensity, 0.0..=4.0),
            "environment_background" => flag(value, &mut environment.show_background),
            "environment_background_blur" => {
                number(value, &mut environment.background_blur, 0.0..=1.0)
            }
            "view_mode" => {
                if let Some(mode) = value
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| ViewMode::ALL.get(index))
                {
                    self.view_mode = *mode;
                }
            }
            "brush_size" => number(value, &mut brush.size, 1.0..=500.0),
            "brush_hardness" => number(value, &mut brush.hardness, 0.0..=1.0),
            "brush_opacity" => number(value, &mut brush.opacity, 0.0..=1.0),
            "brush_paint_color" => flag(value, &mut brush.paint_color),
            "brush_color" => {
                let parts: Vec<f32> = value
                    .split_whitespace()
                    .filter_map(|part| part.parse().ok())
                    .filter(|part: &f32| part.is_finite())
                    .collect();
                if let [r, g, b] = parts[..] {
                    brush.color = [r, g, b].map(|c| c.clamp(0.0, 1.0));
                }
            }
            "brush_paint_roughness" => flag(value, &mut brush.paint_roughness),
            "brush_roughness" => number(value, &mut brush.roughness, 0.0..=1.0),
            "brush_paint_metallic" => flag(value, &mut brush.paint_metallic),
            "brush_metallic" => number(value, &mut brush.metallic, 0.0..=1.0),
            "brush_screen_space" => flag(value, &mut brush.screen_space),
            "brush_paint_over_edges" => flag(value, &mut brush.paint_over_edges),
            _ => {}
        }
    }
}

/// Moves `path` to the front of `recent`, keeping at most `MAX_RECENT_FILES`.
pub fn add_recent_file(recent: &mut Vec<PathBuf>, path: &Path) {
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    recent.retain(|existing| *existing != path);
    recent.insert(0, path);
    recent.truncate(MAX_RECENT_FILES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Channel;

    #[test]
    fn settings_survive_a_round_trip() {
        let mut settings = Settings::default();
        settings.recent_files = vec![
            PathBuf::from("/models/car with spaces.glb"),
            PathBuf::from("/projects/car = final.rustpaint"),
        ];
        settings.environment_source = EnvironmentSource::File(PathBuf::from("/hdri/sky.exr"));
        settings.environment.rotation_degrees = 90.5;
        settings.environment.intensity = 2.0;
        settings.environment.show_background = false;
        settings.environment.background_blur = 0.25;
        settings.view_mode = ViewMode::Channel(Channel::Roughness);
        settings.brush.size = 120.0;
        settings.brush.color = [0.25, 0.5, 1.0];
        settings.brush.paint_metallic = true;
        settings.brush.screen_space = true;

        assert_eq!(Settings::from_text(&settings.to_text()), settings);

        settings.environment_source = EnvironmentSource::BuiltIn(BuiltIn::Studio);
        assert_eq!(Settings::from_text(&settings.to_text()), settings);
    }

    #[test]
    fn bad_lines_fall_back_to_defaults() {
        let settings = Settings::from_text(
            "garbage\nbrush_size = huge\nenvironment = nowhere\nview_mode = 99\n\
             brush_opacity = 7\nbrush_color = 1 2\nunknown_key = 1\nbrush_hardness = NaN\n",
        );
        let mut expected = Settings::default();
        expected.brush.opacity = 1.0; // clamped
        assert_eq!(settings, expected);
    }

    #[test]
    fn recent_files_are_unique_and_newest_first() {
        let mut recent = Vec::new();
        for i in 0..MAX_RECENT_FILES + 3 {
            add_recent_file(&mut recent, Path::new(&format!("/m/{i}.glb")));
        }
        add_recent_file(&mut recent, Path::new("/m/5.glb"));
        assert_eq!(recent.len(), MAX_RECENT_FILES);
        assert_eq!(recent[0], PathBuf::from("/m/5.glb"));
        assert_eq!(
            recent[1],
            PathBuf::from(format!("/m/{}.glb", MAX_RECENT_FILES + 2))
        );
        assert_eq!(recent.iter().filter(|p| p.ends_with("5.glb")).count(), 1);
    }
}
