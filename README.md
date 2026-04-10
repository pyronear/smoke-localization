# Smoke Localisation

Estimate the geographic location of smoke detected in a single frame from a [Pyronear](https://pyronear.org/) wildfire detection camera.

Given a camera image and its mounting parameters, the tool projects the clicked pixel onto the terrain and returns GPS coordinates of the smoke origin.

## How it works

### 1. Sky detection & tilt auto-calibration

Wildfire cameras are mounted on towers or hilltops, and their exact tilt (pitch) is often unknown or approximate. To solve this, we run a sky segmentation model on the uploaded image:

- An **ncnn** model (`skysegsmall_sim-opt-fp16`) produces a binary sky mask
- For each column, the lowest sky pixel defines the **terrain silhouette**
- A line is fitted to the silhouette to find the **horizon position** in the image
- The vertical offset of the horizon from image centre, combined with the known vertical FOV, gives the **camera tilt**

If the clicked pixel falls in the sky region, the app warns the user instead of producing a wrong projection.

### 2. Digital Surface Model (DSM)

The tool downloads [Copernicus DEM GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) tiles (30 m resolution) from AWS. This is a **Digital Surface Model** -- it includes tree canopy and building heights, not just bare ground. Tiles are cached locally in `dem_cache/` (not committed to git).

### 3. Ray-terrain intersection

A pinhole camera model converts the clicked pixel `(u, v)` into a 3D ray:

- **Horizontal angle**: `camera_azimuth + (u - 0.5) * HFOV`
- **Vertical angle**: `(0.5 - v) * VFOV - tilt`

The ray is then marched in 5 m steps from the camera position. At each step, the ray altitude is compared against the DSM elevation (queried via bilinear interpolation). When the ray drops below the terrain surface, a bisection refinement pinpoints the intersection.

## Streamlit app

```
          +------------------+-------------------+
          |  Camera image    |  OpenStreetMap     |
          |  (click to pick  |  - camera marker   |
          |   a pixel)       |  - hit marker      |
          |                  |  - ray line         |
          |  Sky overlay +   |                    |
          |  horizon line    |                    |
          +------------------+-------------------+
          Sidebar: lat/lon, azimuth, FOV, height
```

### Running

```bash
uv run streamlit run app/app.py
```

### Camera parameters (sidebar)

| Parameter | Description |
|---|---|
| Location (lat, lon) | Camera GPS coordinates (WGS84) |
| Azimuth | Direction the camera centre points (degrees, clockwise from north) |
| Horizontal FOV | Horizontal field of view (degrees) |
| Height above ground | Camera height on tower/mast (metres) |
| Auto-calibrate tilt | Use sky detection to estimate tilt (recommended) |
| Manual tilt | Fallback if auto-calibration is disabled (degrees, positive = down) |

Aspect ratio is read automatically from the uploaded image.

## Project structure

```
smoke-localisation/
  src/smoke_localisation/
    __init__.py             Public API
    camera.py               Pinhole camera model, pixel to ray
    dem.py                  DSM download (Copernicus GLO-30), loading, caching
    ray.py                  Ray-terrain intersection (marching + bisection)
    sky.py                  Sky segmentation (ncnn), horizon detection, tilt estimation
    models/skyseg/          ncnn model weights (.param + .bin)
  app/
    app.py                  Streamlit UI
  data/
    cameras.csv             Camera registry (name, lat, lon, fov, height)
    samples/                Demo images (source_site_azimuth_date.jpg)
  tests/                    Unit tests
  dem_cache/                Downloaded DEM tiles (git-ignored)
  pyproject.toml            Dependencies managed by uv
```

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/). All installed automatically on first run.

- `ncnn` -- sky segmentation inference
- `opencv-python-headless` -- image I/O for the sky detector
- `rasterio` -- reads GeoTIFF DSM tiles
- `pyproj` -- geodesic forward computation (camera to ground point)
- `scipy` -- bilinear interpolation on the DSM grid
- `streamlit`, `folium`, `streamlit-folium`, `streamlit-image-coordinates` -- web UI

## Limitations & investigation topics

### Current limitations

- Pinhole camera model (no lens distortion correction)
- DSM resolution is 30 m -- individual small structures are not resolved
- Sky-based tilt calibration assumes the terrain silhouette is close to the geometric horizon; in very mountainous scenes the estimated tilt may be slightly off
- Atmospheric refraction is not modelled

### Investigations to improve accuracy

**Camera model**

- [ ] **Lens distortion correction** -- wide-angle cameras (FOV > 60) have significant barrel distortion. Applying a radial distortion model (Brown-Conrady or fisheye) before projecting would improve accuracy at image edges. Requires calibration coefficients per camera model, or estimation from straight lines in the image.
- [ ] **Per-camera intrinsics from EXIF** -- extract focal length and sensor size from image EXIF metadata to compute FOV automatically instead of requiring manual input.
- [ ] **Roll estimation** -- the current model assumes the camera is level (no roll). A tilted horizon in the sky mask could be used to estimate and correct roll angle.

**Tilt calibration**

- [ ] **Terrain silhouette vs geometric horizon** -- the sky boundary includes hills and trees, which sit above the true 0-elevation horizon. Investigate using the DSM itself to render the expected terrain silhouette for the given camera position/azimuth, then match it against the detected sky boundary to jointly refine tilt and azimuth.
- [ ] **DSM-based horizon rendering** -- for a known camera position, render the expected skyline from the DSM and cross-correlate with the detected sky mask. This would give a precise tilt+azimuth fit without relying on the geometric horizon assumption.
- [ ] **Multi-image tilt averaging** -- use several frames (different times of day, weather) to average out noise in the sky detection and get a more robust tilt estimate.

**Terrain & projection**

- [ ] **Higher resolution DSM** -- investigate using IGN's RGE ALTI (1 m or 5 m resolution, France-specific) or LiDAR-derived DSMs for better precision, especially for nearby hits where 30 m resolution matters.
- [ ] **Atmospheric refraction correction** -- for long-range projections (> 10 km), atmospheric refraction bends the ray downward. A standard refraction model (e.g. 4/3 Earth radius approximation) would improve distant hits.
- [ ] **Confidence / uncertainty estimation** -- the ray angle depends on pixel position, tilt, and FOV. Propagating small uncertainties on these inputs would give a confidence ellipse on the ground hit instead of a single point.

**Operational**

- [ ] **Batch mode / API** -- accept detection coordinates from the Pyronear alert pipeline directly (JSON with camera ID, bounding box) instead of manual clicking.
- [ ] **Multi-camera triangulation** -- when two cameras see the same smoke, intersecting their rays would give a much more precise 3D location and remove the dependency on DSM accuracy.
