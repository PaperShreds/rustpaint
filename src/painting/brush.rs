#[derive(Clone, Debug, PartialEq)]
pub struct Brush {
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
    /// Remove paint from the layer instead of adding it.
    pub erase: bool,
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
            erase: false,
        }
    }
}
