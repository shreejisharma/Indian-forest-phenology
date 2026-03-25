// =====================================================================
// MODIS + SENTINEL-2 + LANDSAT 8&9 — DATE + NDVI — TIRUPATI
// Single script → 3 separate CSV exports
// =====================================================================

var SITE_NAME = 'tirupati';
var LATITUDE  = 13.63;
var LONGITUDE = 79.42;
var BUFFER_M  = 5000;

// ── DATE RANGE ────────────────────────────────────────────────────────
// Start dates are set to each sensor's first ever available date
// End date fixed to March 2026
var END_DATE = '2026-03-31';
// ─────────────────────────────────────────────────────────────────────

var point = ee.Geometry.Point([LONGITUDE, LATITUDE]);
var aoi   = point.buffer(BUFFER_M);

// ── SENSOR FIRST-EVER AVAILABLE DATES (auto start) ───────────────────
var MODIS_TERRA_START = '2000-02-18';   // MOD13Q1 first composite
var MODIS_AQUA_START  = '2002-07-04';   // MYD13Q1 first composite
var S2_START          = '2015-06-23';   // Sentinel-2A first scene
var L8_START          = '2013-04-11';   // Landsat 8 OLI first scene
var L9_START          = '2021-10-31';   // Landsat 9 OLI-2 first scene

// =====================================================================
// 1. MODIS (Terra + Aqua) | 2000-02-18 → 2026-03-31
// =====================================================================
function prepMODIS(img) {
  var mask = img.select('SummaryQA').lte(1);
  var ndvi = img.select('NDVI').multiply(0.0001).rename('NDVI');
  return img.addBands(ndvi, null, true)
            .updateMask(mask)
            .copyProperties(img, ['system:time_start']);
}

var terra = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(MODIS_TERRA_START, END_DATE)
  .filterBounds(aoi)
  .map(prepMODIS);

var aqua = ee.ImageCollection('MODIS/061/MYD13Q1')
  .filterDate(MODIS_AQUA_START, END_DATE)
  .filterBounds(aoi)
  .map(prepMODIS);

var modis = terra.merge(aqua).sort('system:time_start');

var modisTS = modis.map(function(img) {
  var val = img.select('NDVI').reduceRegion({
    reducer  : ee.Reducer.mean(),
    geometry : aoi,
    scale    : 250,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'NDVI': val.get('NDVI')
  });
});

// =====================================================================
// 2. SENTINEL-2 | 2015-06-23 → 2026-03-31
// =====================================================================
function maskS2(img) {
  var scl  = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return img.updateMask(mask).divide(10000)
            .copyProperties(img, ['system:time_start']);
}

function addNDVI_S2(img) {
  return img.addBands(
    img.select('B8').subtract(img.select('B4'))
       .divide(img.select('B8').add(img.select('B4')))
       .rename('NDVI')
  );
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(S2_START, END_DATE)
  .filterBounds(aoi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
  .map(maskS2)
  .map(addNDVI_S2)
  .sort('system:time_start');

var s2TS = s2.map(function(img) {
  var val = img.select('NDVI').reduceRegion({
    reducer  : ee.Reducer.mean(),
    geometry : aoi,
    scale    : 10,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'NDVI': val.get('NDVI')
  });
});

// =====================================================================
// 3. LANDSAT 8 + 9 | 2013-04-11 → 2026-03-31
// =====================================================================
function maskL(img) {
  var qa   = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
               .and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.updateMask(mask).divide(10000)
            .copyProperties(img, ['system:time_start']);
}

function addNDVI_L(img) {
  return img.addBands(
    img.select('SR_B5').subtract(img.select('SR_B4'))
       .divide(img.select('SR_B5').add(img.select('SR_B4')))
       .rename('NDVI')
  );
}

function prepL(col) {
  return col.filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUD_COVER', 80))
            .map(maskL)
            .map(addNDVI_L);
}

var landsat = prepL(
  ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(L8_START, END_DATE)
).merge(
  prepL(
    ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
      .filterDate(L9_START, END_DATE)
  )
).sort('system:time_start');

var landsatTS = landsat.map(function(img) {
  var val = img.select('NDVI').reduceRegion({
    reducer  : ee.Reducer.mean(),
    geometry : aoi,
    scale    : 30,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
    'NDVI': val.get('NDVI')
  });
});

// =====================================================================
// PRINT SCENE COUNTS
// =====================================================================
print('── Scene counts ──────────────────────────────');
print('MODIS Terra  (2000-02-18 → 2026-03-31) :', terra.size());
print('MODIS Aqua   (2002-07-04 → 2026-03-31) :', aqua.size());
print('MODIS Total                            :', modis.size());
print('Sentinel-2   (2015-06-23 → 2026-03-31) :', s2.size());
print('Landsat 8+9  (2013-04-11 → 2026-03-31) :', landsat.size());

// =====================================================================
// CHARTS
// =====================================================================
print('── Charts ────────────────────────────────────');

print(ui.Chart.feature.byFeature(modisTS, 'date', ['NDVI'])
  .setChartType('LineChart')
  .setOptions({
    title    : 'MODIS NDVI — Tirupati (2000 → Mar 2026)',
    hAxis    : {title: 'Date'},
    vAxis    : {title: 'NDVI', viewWindow: {min: -0.2, max: 1}},
    lineWidth: 1, pointSize: 2, colors: ['#e65100']
  }));

print(ui.Chart.feature.byFeature(s2TS, 'date', ['NDVI'])
  .setChartType('LineChart')
  .setOptions({
    title    : 'Sentinel-2 NDVI — Tirupati (2015 → Mar 2026)',
    hAxis    : {title: 'Date'},
    vAxis    : {title: 'NDVI', viewWindow: {min: -0.2, max: 1}},
    lineWidth: 1, pointSize: 2, colors: ['#1565c0']
  }));

print(ui.Chart.feature.byFeature(landsatTS, 'date', ['NDVI'])
  .setChartType('LineChart')
  .setOptions({
    title    : 'Landsat 8+9 NDVI — Tirupati (2013 → Mar 2026)',
    hAxis    : {title: 'Date'},
    vAxis    : {title: 'NDVI', viewWindow: {min: -0.2, max: 1}},
    lineWidth: 1, pointSize: 3, colors: ['#2e7d32']
  }));

// =====================================================================
// 3 EXPORT TASKS
// =====================================================================
print('── Exports (check Tasks tab) ─────────────────');

Export.table.toDrive({
  collection : ee.FeatureCollection(modisTS),
  description: SITE_NAME + '_MODIS_NDVI_2000_to_Mar2026',
  folder     : 'GEE_exports',
  fileFormat : 'CSV',
  selectors  : ['date', 'NDVI']
});
print('1. tirupati_MODIS_NDVI_2000_to_Mar2026.csv');

Export.table.toDrive({
  collection : ee.FeatureCollection(s2TS),
  description: SITE_NAME + '_S2_NDVI_2015_to_Mar2026',
  folder     : 'GEE_exports',
  fileFormat : 'CSV',
  selectors  : ['date', 'NDVI']
});
print('2. tirupati_S2_NDVI_2015_to_Mar2026.csv');

Export.table.toDrive({
  collection : ee.FeatureCollection(landsatTS),
  description: SITE_NAME + '_L8L9_NDVI_2013_to_Mar2026',
  folder     : 'GEE_exports',
  fileFormat : 'CSV',
  selectors  : ['date', 'NDVI']
});
print('3. tirupati_L8L9_NDVI_2013_to_Mar2026.csv');

print('✓ Go to Tasks tab → click Run on all 3 tasks');
