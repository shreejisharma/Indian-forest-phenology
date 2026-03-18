/**
 * ═══════════════════════════════════════════════════════════════
 * Google Earth Engine Script — MODIS MOD13Q1 NDVI Extraction
 * Indian Forest Phenology Predictor
 * ═══════════════════════════════════════════════════════════════
 */

// ── CONFIGURATION — edit these values ─────────────────────────
var SITE_NAME  = 'Tirupati';
var LATITUDE   = 13.63;
var LONGITUDE  = 79.42;
var BUFFER_M   = 500;
var START_DATE = '2016-01-01';
var END_DATE   = '2025-12-31';
// ───────────────────────────────────────────────────────────────

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var roi   = point.buffer(BUFFER_M);

var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roi)
  .select(['NDVI', 'SummaryQA']);  // ← FIXED: was 'pixel_reliability'

// Quality filter: keep only good (0) and marginal (1) pixels
var filtered = modis.map(function(img) {
  var qa = img.select('SummaryQA');  // ← FIXED
  return img.updateMask(qa.lte(1));
});

// Extract mean NDVI per composite, apply scale factor 0.0001
var timeSeries = filtered.map(function(img) {
  var ndvi = img.select('NDVI').multiply(0.0001);
  var val  = ndvi.reduceRegion({
    reducer:   ee.Reducer.mean(),
    geometry:  roi,
    scale:     250,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date': img.date().format('YYYY-MM-dd'),
    'NDVI': val.get('NDVI')
  });
});

// Remove observations with no valid NDVI (complete cloud cover)
var clean = ee.FeatureCollection(timeSeries)
  .filter(ee.Filter.notNull(['NDVI']));

// Export to Google Drive
Export.table.toDrive({
  collection:  clean,
  description: 'MODIS_NDVI_' + SITE_NAME + '_' + START_DATE.slice(0,4) + '_' + END_DATE.slice(0,4),
  fileFormat:  'CSV',
  selectors:   ['date', 'NDVI']
});

print('Total observations:', clean.size());
print('Sample (first 5):', clean.limit(5));
