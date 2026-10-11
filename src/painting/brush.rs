/// Brush diameter limits, in screen pixels. The size slider stops at
/// `SLIDER_MAX_SIZE`; zooming with `keep_size_on_model` can go further.
pub const MIN_SIZE: f32 = 1.0;
pub const SLIDER_MAX_SIZE: f32 = 512.0;
pub const MAX_SIZE: f32 = 2048.0;

/// What a click or drag in the viewport does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Paint,
    /// Push the layer's paint along the stroke, like a finger in wet paint.
    Smear,
    /// Soften the layer's paint under the brush.
    Blur,
    /// Fill an area of the layer with the brush's values in one click.
    Fill,
}

impl Tool {
    pub const ALL: [Tool; 4] = [Tool::Paint, Tool::Smear, Tool::Blur, Tool::Fill];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Paint => "Paint",
            Tool::Smear => "Smear",
            Tool::Blur => "Blur",
            Tool::Fill => "Fill",
        }
    }

    /// Whether the erase toggle applies (it removes paint instead of adding it).
    pub fn can_erase(self) -> bool {
        matches!(self, Tool::Paint | Tool::Fill)
    }
}

/// How much of the layer the fill tool covers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FillMode {
    /// Every material slot the layer paints on.
    #[default]
    Layer,
    /// The material slot under the cursor.
    Material,
    /// The UV island under the cursor: triangles joined by shared UV edges.
    UvIsland,
}

impl FillMode {
    pub const ALL: [FillMode; 3] = [FillMode::Layer, FillMode::Material, FillMode::UvIsland];

    pub fn name(self) -> &'static str {
        match self {
            FillMode::Layer => "Whole layer",
            FillMode::Material => "Clicked material",
            FillMode::UvIsland => "Clicked UV island",
        }
    }

    /// Stable name for the settings file.
    pub fn key(self) -> &'static str {
        match self {
            FillMode::Layer => "layer",
            FillMode::Material => "material",
            FillMode::UvIsland => "uv_island",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.key() == key)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Brush {
    /// Diameter in screen pixels.
    pub size: f32,
    pub hardness: f32,
    pub opacity: f32,
    /// Which channels the brush paints, and what it puts there.
    pub paint_color: bool,
    pub color: [f32; 3],
    pub paint_roughness: bool,
    pub roughness: f32,
    pub paint_metallic: bool,
    pub metallic: f32,
    /// Measure the brush footprint in screen space instead of along the surface.
    pub screen_space: bool,
    /// Let a stroke continue across hard edges, instead of staying on the
    /// face (the area bounded by hard edges) where it started.
    pub paint_over_edges: bool,
    /// Give the surface brush one size on the model, measured at the depth
    /// of the orbit centre and scaled with zooming. Otherwise it keeps its
    /// size on screen, so it covers more of the model further away.
    pub keep_size_on_model: bool,
    pub tool: Tool,
    /// Paint and Fill remove paint from the layer instead of adding it.
    pub erase: bool,
    pub fill_mode: FillMode,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            size: 32.0,
            hardness: 0.8,
            opacity: 1.0,
            paint_color: true,
            color: [1.0, 1.0, 1.0],
            paint_roughness: false,
            roughness: 0.5,
            paint_metallic: false,
            metallic: 0.0,
            screen_space: false,
            paint_over_edges: true,
            keep_size_on_model: false,
            tool: Tool::Paint,
            erase: false,
            fill_mode: FillMode::Layer,
        }
    }
}

impl Brush {
    /// Whether strokes remove paint: the erase toggle, for tools it applies to.
    pub fn erases(&self) -> bool {
        self.erase && self.tool.can_erase()
    }
}
