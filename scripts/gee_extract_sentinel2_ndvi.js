/**
 * ═══════════════════════════════════════════════════════════════
 * Google Earth Engine Script — Sentinel-2 Monthly NDVI Extraction
 * Indian Forest Phenology Predictor
 * ═══════════════════════════════════════════════════════════════
 *
 * USAGE:
 *   1. Open https://code.earthengine.google.com
 *   2. Paste this script
 *   3. Edit CONFIGURATION section
 *   4. Click Run → Tasks → Run export
 *
 * OUTPUT: CSV with monthly NDVI + NDSI + NDWI + n_scenes
 * SENSOR: Sentinel-2 SR Harmonized (10m)
 * CLOUD MASK: SCL-based (removes cloud, shadow, snow classes)
 *
 * RECOMMENDED FOR:
 *   Alpine / Subalpine (Spiti, Valley of Flowers)
 *   Tropical Thorn Scrub (Jaisalmer)
 *   Subtropical Hill Forest
 * ═══════════════════════════════════════════════════════════════
 */

// ── CONFIGURATION — edit these values ─────────────────────────
var SITE_NAME  = 'Spiti';           // used in output filename
var LATITUDE   = 32.24;             // decimal degrees North
var LONGITUDE  = 78.07;             // decimal degrees East
var BUFFER_M   = 300;               // buffer radius in metres
var START_DATE = '2016-01-01';
var END_DATE   = '2025-12-31';
var SEASON     = 'May-Oct';         // label only (informational)
// ───────────────────────────────────────────────────────────────

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var roi   = point.buffer(BUFFER_M);

// SCL cloud mask: remove cloud shadow(3), medium/high cloud(8,9),
// cirrus(10), snow/ice(11)
function maskS2clouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))
                .and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(mask);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roi)
  .map(maskS2clouds);

// Get distinct year-months to build monthly composites
var months = ee.List.sequence(0, ee.Date(END_DATE).difference(ee.Date(START_DATE), 'month').round());

var timeSeries = months.map(function(offset) {
  var start  = ee.Date(START_DATE).advance(offset, 'month');
  var end    = start.advance(1, 'month');
  var subset = s2.filterDate(start, end);
  var n      = subset.size();

  var composite = subset.median();

  // Band calculations
  var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndsi = composite.normalizedDifference(['B3', 'B11']).rename('NDSI');
  var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');

  var vals = ndvi.addBands(ndsi).addBands(ndwi).reduceRegion({
    reducer:   ee.Reducer.mean(),
    geometry:  roi,
    scale:     10,
    maxPixels: 1e9
  });

  return ee.Feature(null, {
    'date':     start.format('YYYY-MM-dd'),
    'year':     start.get('year'),
    'month':    start.get('month'),
    'NDVI':     vals.get('NDVI'),
    'NDSI':     vals.get('NDSI'),
    'NDWI':     vals.get('NDWI'),
    'n_scenes': n,
    'season':   SEASON,
    'sensor':   'Sentinel2_SR_Harmonized_10m',
    'site_key': SITE_NAME
  });
});

// Remove months with no valid pixels
var clean = ee.FeatureCollection(timeSeries)
  .filter(ee.Filter.notNull(['NDVI']))
  .filter(ee.Filter.gt('n_scenes', 0));

Export.table.toDrive({
  collection:  clean,
  description: 'S2_monthly_' + SITE_NAME + '_' + START_DATE.slice(0,4) + '_' + END_DATE.slice(0,4),
  fileFormat:  'CSV'
});

print('Total monthly composites:', clean.size());

// ── IMPORTANT POST-PROCESSING NOTE ────────────────────────────
// After downloading:
//  - For Alpine sites (Spiti, Valley of Flowers):
//    Remove rows where NDSI > 0.40 OR NDVI < 0.01 (snow-covered)
//    Remove 2016-2017 data if Jun-Sep observations are missing
//  - For Thorn Scrub (Jaisalmer):
//    Do NOT remove low NDVI rows — sparse desert pixels are real
//    Set SOS threshold to 10-12% in the app
// ──────────────────────────────────────────────────────────────

//Sentinel-2 SR NDVI Extraction (Recommended for spatial mapping)//

// ── ROI: Tirupati Forest (~7 km × 4 km) ────────────────────────────────
var region = ee.Geometry.Polygon([[
  [79.58161570026698, 13.70796566350642],
  [79.58320489301980, 13.72528174821274],
  [79.55022411157242, 13.72405112331491],
  [79.55323907125492, 13.70783405528032],
  [79.58161570026698, 13.70796566350642]
]]);

Map.centerObject(region, 13);
Map.addLayer(region, {color: 'red'}, 'Tirupati ROI');


// ── Sentinel-2 SR NDVI collection ──────────────────────────────────────
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(region)
  .filterDate('2017-04-01', '2025-03-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(function(img) {
    // SCL cloud masking: remove cloud shadow (3), medium cloud (8),
    // high cloud (9), thin cirrus (10), snow (11)
    var scl  = img.select('SCL');
    var mask = scl.neq(3)
                  .and(scl.neq(8))
                  .and(scl.neq(9))
                  .and(scl.neq(10))
                  .and(scl.neq(11));
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    return ndvi.updateMask(mask)
               .copyProperties(img, ['system:time_start']);
  });

print('Total S2 NDVI images (2017–2025):', s2.size());


// ── One growing season stack (2024-04-01 to 2025-03-31) ────────────────
var start    = '2024-04-01';
var end      = '2025-03-31';
var ndviYear = s2.filterDate(start, end).sort('system:time_start');

print('Images in 2024–2025 phenology year:', ndviYear.size());

// Get native 10 m scale from first image
var proj  = s2.first().projection();
var scale = proj.nominalScale();  // ~10 m
print('Scale (m):', scale);

// Stack all bands — each band = one acquisition date
var ndviStack = ndviYear.toBands().clip(region);


// ── Export stacked GeoTIFF ──────────────────────────────────────────────
Export.image.toDrive({
  image:          ndviStack,
  description:    'S2_NDVI_Stack_2024_2025_Tirupati',
  fileNamePrefix: 'S2_NDVI_Stack_2024_2025_Tirupati',
  folder:         'GEE_Exports',
  region:         region,
  scale:          scale,
  maxPixels:      1e9,
  fileFormat:     'GeoTIFF'
});


// ── Optional: Time-series point export for temporal analysis ────────────
var pt = ee.Geometry.Point([79.5856, 13.624]);

Export.table.toDrive({
  collection:     s2.getRegion(pt, 100),
  description:   'S2_NDVI_Timeseries_Tirupati',
  fileNamePrefix: 'S2_NDVI_Timeseries_Tirupati',
  folder:         'GEE_Exports',
  fileFormat:     'CSV'
});
