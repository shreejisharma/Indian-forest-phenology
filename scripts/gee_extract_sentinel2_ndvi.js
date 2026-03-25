/**
 * ═══════════════════════════════════════════════════════════════
 * Google Earth Engine Script — Sentinel-2 5-DAY NDVI (SENSOR NATIVE)
 * Indian Forest Phenology Predictor — ALL ERRORS FIXED
 * ═══════════════════════════════════════════════════════════════
 */

var SITE_NAME     = 'tirupati';
var LATITUDE      = 13.63;
var LONGITUDE     = 79.42;
var BUFFER_M      = 300;
var START_DATE    = '2016-01-01';
var END_DATE      = '2025-12-31';
var INTERVAL_DAYS = 5;
var SEASON        = 'Apr-Mar';

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var roi   = point.buffer(BUFFER_M);

function maskS2clouds(img) {
  var scl = img.select('SCL');
  var unwanted = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10)).or(scl.eq(11));
  var qa60 = img.select('QA60');
  var aerosol = qa60.bitwiseAnd(1 << 10).neq(0);
  var mask = unwanted.not().and(aerosol.not());
  var bands = img.select('B2', 'B3', 'B4', 'B8', 'B11');
  return bands.updateMask(mask).copyProperties(img, ['system:time_start']);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
  .filterBounds(roi)
  .map(maskS2clouds);

print('Total scenes:', s2.size());

var totalDays = ee.Date(END_DATE).difference(ee.Date(START_DATE), 'day').round();
var periods = ee.List.sequence(0, totalDays.divide(INTERVAL_DAYS).round());

var timeSeries = periods.map(function(offset) {
  offset = ee.Number(offset);
  
  var start = ee.Date(START_DATE).advance(offset.multiply(INTERVAL_DAYS), 'day');
  var end   = start.advance(INTERVAL_DAYS, 'day');
  var subset = s2.filterDate(start, end);
  var n = subset.size();
  
  var composite = ee.Algorithms.If(n.gt(0), subset.median(), ee.Image([]));
  composite = ee.Image(composite);
  
  var hasBands = composite.bandNames().size().gt(0);
  
  var ndvi = ee.Algorithms.If(hasBands, 
    composite.normalizedDifference(['B8', 'B4']).rename('NDVI'), 
    ee.Image(0).rename('NDVI'));
    
  var ndsi = ee.Algorithms.If(hasBands, 
    composite.normalizedDifference(['B3', 'B11']).rename('NDSI'), 
    ee.Image(0).rename('NDSI'));
    
  var ndwi = ee.Algorithms.If(hasBands, 
    composite.normalizedDifference(['B3', 'B8']).rename('NDWI'), 
    ee.Image(0).rename('NDWI'));
    
  var indices = ee.Image(ndvi).addBands(ee.Image(ndsi)).addBands(ee.Image(ndwi));
  
  var vals = indices.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 10,
    maxPixels: 1e9
  });

  return ee.Feature(null, {
    'date': start.format('YYYY-MM-dd'),
    'doy': start.getRelative('day', 'year'),  // ← FIXED: Proper DOY (0-365)
    'year': start.get('year'),
    'month': start.get('month'),
    'interval': offset,
    'NDVI': vals.get('NDVI'),
    'NDSI': vals.get('NDSI'),
    'NDWI': vals.get('NDWI'),
    'n_scenes': n,
    'season': SEASON,
    'sensor': 'Sentinel2_5day_10m',
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
  description: 'S2_5day_' + SITE_NAME + '_' + START_DATE.slice(0,4) + '_to_' + END_DATE.slice(0,4),
  fileFormat: 'CSV'
});

print('Total 5-day periods:', periods.size());
print('Valid records exported:', clean.size());
print('Sample:', clean.limit(10));

Map.centerObject(roi, 14);
Map.addLayer(roi, {color: 'red'}, 'ROI 300m');
