pub struct Brush {
    pub size: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub color: [f32; 3],
    /// Measure the brush footprint in screen space instead of along the surface.
    pub screen_space: bool,
    /// Let a stroke continue across hard edges, instead of staying on the
    /// face (the area bounded by hard edges) where it started.
    pub paint_over_edges: bool,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            size: 32.0,
            hardness: 0.8,
            opacity: 1.0,
            color: [1.0, 1.0, 1.0],
            screen_space: false,
            paint_over_edges: true,
        }
    }
}
