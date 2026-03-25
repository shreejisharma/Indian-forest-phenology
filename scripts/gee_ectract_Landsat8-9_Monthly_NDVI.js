/**
 * ═══════════════════════════════════════════════════════════════
 * Google Earth Engine Script — Landsat 8/9 8-DAY NDVI Extraction
 * Indian Forest Phenology Predictor — SENSOR NATIVE INTERVAL
 * ═══════════════════════════════════════════════════════════════
 *
 * OUTPUT: CSV with 8-day NDVI + NDSI + NDWI + n_scenes (30m)
 * INTERVAL: Every 8 days (Landsat 8+9 native temporal resolution)
 */

var SITE_NAME     = 'TDD';
var LATITUDE      = 13.63;
var LONGITUDE     = 79.42;
var BUFFER_M      = 300;
var START_DATE    = '2015-01-01';
var END_DATE      = '2025-12-31';
var INTERVAL_DAYS = 8;    // ← LANDSAT NATIVE: 8-day revisit
var SEASON        = 'Apr-Mar';

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var roi   = point.buffer(BUFFER_M);

function maskLandsatSR(img) {
  var qa = img.select('QA_PIXEL');
  var dilatedCloud = qa.bitwiseAnd(1 << 1).neq(0);
  var cloud        = qa.bitwiseAnd(1 << 3).neq(0);
  var cloudShadow  = qa.bitwiseAnd(1 << 4).neq(0);
  var snow         = qa.bitwiseAnd(1 << 5).neq(0);
  var mask         = dilatedCloud.or(cloud).or(cloudShadow).or(snow).not();
  
  var srBands = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']);  // Key bands only
  return srBands.updateMask(mask).copyProperties(img, ['system:time_start']);
}

var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roi)
  .map(maskLandsatSR);

var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roi)
  .map(maskLandsatSR);

var landsat = L8.merge(L9);

function applyScaleFactors(img) {
  var hasBands = img.bandNames().size().gt(0);
  var opticalBands = ee.Algorithms.If(
    hasBands,
    img.multiply(0.0000275).add(-0.2).clamp(0, 1),
    img
  );
  return ee.Image(opticalBands).copyProperties(img, ['system:time_start']);
}

landsat = landsat.map(applyScaleFactors);

print('Total Landsat scenes:', landsat.size());

// ── 8-DAY INTERVALS (LANDSAT NATIVE) ───────────────────────────
var totalDays = ee.Date(END_DATE).difference(ee.Date(START_DATE), 'day').round();
var periods = ee.List.sequence(0, totalDays.divide(INTERVAL_DAYS).round());

var timeSeries = periods.map(function(offset) {
  offset = ee.Number(offset);
  
  var start = ee.Date(START_DATE).advance(offset.multiply(INTERVAL_DAYS), 'day');
  var end   = start.advance(INTERVAL_DAYS, 'day');
  var subset = landsat.filterDate(start, end);
  var n = subset.size();
  
  var composite = ee.Algorithms.If(n.gt(0), subset.median(), ee.Image([]));
  composite = ee.Image(composite);
  
  var hasBands = composite.bandNames().size().gt(0);
  
  var ndvi = ee.Algorithms.If(hasBands,
    composite.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'),
    ee.Image(0).rename('NDVI')
  );
  var ndsi = ee.Algorithms.If(hasBands,
    composite.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI'),
    ee.Image(0).rename('NDSI')
  );
  var ndwi = ee.Algorithms.If(hasBands,
    composite.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI'),
    ee.Image(0).rename('NDWI')
  );

  var indices = ee.Image(ndvi).addBands(ee.Image(ndsi)).addBands(ee.Image(ndwi));
  
  var vals = indices.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 30,
    maxPixels: 1e9
  });

  return ee.Feature(null, {
    'date': start.format('YYYY-MM-dd'),
    'doy': start.getRelative('day', 'year'),  // Proper DOY (0-365)
    'year': start.get('year'),
    'month': start.get('month'),
    'interval': offset,
    'NDVI': vals.get('NDVI'),
    'NDSI': vals.get('NDSI'),
    'NDWI': vals.get('NDWI'),
    'n_scenes': n,
    'season': SEASON,
    'sensor': 'Landsat8_9_8day_30m',
    'days_since_start': offset.multiply(INTERVAL_DAYS),
    'site_key': SITE_NAME
  });
});

var clean = ee.FeatureCollection(timeSeries)
  .filter(ee.Filter.notNull(['NDVI']))
  .filter(ee.Filter.gt('n_scenes', 0))
  .sort('date');

Export.table.toDrive({
  collection: clean,
  description: 'L89_8day_' + SITE_NAME + '_' + START_DATE.slice(0,4) + '_to_' + END_DATE.slice(0,4),
  fileFormat: 'CSV'
});

print('Total 8-day periods:', periods.size());
print('Valid Landsat records:', clean.size());
print('Sample:', clean.limit(10));

Map.centerObject(roi, 14);
Map.addLayer(roi, {color: 'blue'}, 'Landsat ROI');
